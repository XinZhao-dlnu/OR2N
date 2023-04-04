import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

""" Meta Module"""
""" --------------------------------------------------------------------"""


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # print(name_t)
                # if name_t.split('.')[0] not in ['feature', 'feature_fuse_1', 'feature_fuse_2', 'bh1d_1', 'bh1d_2',
                #                                 'fc6', 'fc7']:
                #     continue
                if src is not None:
                    # name_s, param_s = src
                    # grad = param_s.grad
                    # name_s, param_s = src
                    grad = src
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param_t - lr_inner * grad
                    self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super(MetaLinear, self).__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super(MetaConv2d, self).__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super(MetaConvTranspose2d, self).__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super(MetaBatchNorm2d, self).__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(MetaBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


import numbers


class MetaLayerNorm(MetaModule):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MetaLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            ignore_w = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            ignore_bias = nn.Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.register_buffer('weight', to_var(ignore_w.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore_bias.data, requires_grad=True))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, input):
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm1d(MetaModule):
    def __init__(self, *args, **kwargs):
        super(MetaBatchNorm1d, self).__init__()
        ignore = nn.BatchNorm1d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(MetaBatchNorm1d, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')  # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


"""Relation Network"""
""" --------------------------------------------------------------------"""


class RelationNetwork(MetaModule):
    def __init__(self, feature_dim):
        super(KinRelation, self).__init__()

        self.part = 4  # We cut the pool5 to p parts
        # self.part_avgpool = nn.AdaptiveMaxPool2d((self.part, 1))
        for i in range(self.part):
            name = 'classifier' + str(i)
            setattr(self, name, ClassBlock(256, 502, droprate=0.5))  # 502, 513

        self.layer_1 = nn.Sequential(
            MetaConv2d(3, 64, kernel_size=3, padding=1),
            MetaBatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer_2 = nn.Sequential(
            MetaConv2d(64, 64, kernel_size=3, padding=1),
            MetaBatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer_3 = nn.Sequential(
            MetaConv2d(64, 64, kernel_size=3, padding=1),
            MetaBatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )
        self.layer_4 = nn.Sequential(
            MetaConv2d(64, 64, kernel_size=3, padding=0),
            MetaBatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU()
        )

        self.bn2d_1 = MetaBatchNorm2d(256)

        self.layer_5 = nn.Sequential(
            MetaConv2d(512, 64, kernel_size=3, padding=0),
            MetaBatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer_6 = nn.Sequential(
            MetaConv2d(64, 64, kernel_size=3, padding=0),
            MetaBatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_1 = MetaLinear(2304, 64)
        self.fc_2 = MetaLinear(64, 1)

    def forward(self, x1, x2):
        batch_size = x1.size()[0]

        '''part_based family id'''
        x1 = self.forwardonce(x1)
        x2 = self.forwardonce(x2)  # B, 256, 32, 32

        '''part_based'''
        # part-based family ID losses
        family_id = torch.cat((x1, x2), dim=0)  # 2B, 256, 32, 32
        family_id = F.avg_pool2d(family_id, kernel_size=(32//self.part, family_id.size(3)),
                                 stride=(32//self.part, family_id.size(3)))  # 2B, 256, p, 1
        family_id = self.bn2d_1(family_id)
        family_id = F.relu(family_id)

        part = {}
        predict = {}
        # family_id = []
        family_out = []
        # get 3 parts feature bs*2048*3
        for i in range(self.part):
            part[i] = torch.squeeze(family_id[:, :, i])  # 2B, 256
            name = 'classifier' + str(i)
            c = getattr(self, name)
            predict[i] = c(part[i])  # 2B, 150
            family_out.append(predict[i])

        '''kin/no kin'''
        kin = torch.cat((x1, x2), dim=1)
        out = self.layer_5(kin)
        out = self.layer_6(out)
        # print(out.size())
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = torch.sigmoid(self.fc_2(out).view(batch_size, ))

        return out, family_out

    def forwardonce(self, x):
        out1 = self.layer_1(x)
        out2 = self.layer_2(out1)
        out3 = self.layer_3(out2)
        out4 = self.layer_4(out3)

        out2 = F.pad(out2, [8, 8, 8, 8])
        out3 = F.pad(out3, [8, 8, 8, 8])
        out4 = F.pad(out4, [9, 9, 9, 9])

        out = torch.cat([out1, out2, out3, out4], dim=1)

        return out


class MetaNet(MetaModule):
    def __init__(self):
        super(MetaNet, self).__init__()
        self.linear1 = MetaLinear(1, 256)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(256, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return F.sigmoid(x)



