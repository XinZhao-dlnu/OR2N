import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

import aveMeter
from dataloader import test_dataloader, train_pos_dataloader, train_neg_dataloader, meta_data_loader
from Rnet import RelationNetwork, MetaNet
from focal_loss import FocalLoss

parser = argparse.ArgumentParser(description='PyTorch Kinship')
parser.add_argument('--batch-c', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 2)')
parser.add_argument('--part', type=int, default=4, metavar='N',
                    help='split numbers for training (default: 4)')
parser.add_argument('--lr-decay-epoch', type=str, default='100',
                    help='epochs at which learning rate decays. default is 100,150.')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--valid-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for validation (default: 64)')
parser.add_argument('--save-model', type=str, default='model/',
                    help='where you save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr2', type=float, default=1e-4, metavar='LR2',
                    help='learning rate 2')
parser.add_argument('--meta-lr', type=float, default=5e-4, metavar='MLR',
                    help='meta learning rate')
parser.add_argument('--stop-epoch', type=int, default=100, metavar='N',
                    help='number of stage-I to train (default: 100)')
parser.add_argument('--max-epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--print-freq', default=5, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--manualSeed', type=int, default=-1,
                    help='manual seed')
parser.add_argument('--num-workers', default=4, type=int,
                    help='number of load data workers (default: 4)')
parser.add_argument('--relat', default="fd", type=str,
                    help='relationship among 4 classes (default: md)')
parser.add_argument('--log', type=str, default='log/',
                    help='where you save log file')
parser.add_argument('--data-root', type=str, default='./data/',
                    help='data root')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.manualSeed is None or args.manualSeed < 0:
    args.manualSeed = random.randint(1, 10000)

if not os.path.exists(args.log):
    os.mkdir(args.log)
if not os.path.exists(args.log + args.relat):
    os.mkdir(args.log + args.relat)
if not os.path.exists(args.save_model):
    os.mkdir(args.save_model)

torch.set_num_threads(1)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(args.manualSeed)


def acc_run(kin_model, validloader):
    evaluate_result = []
    correct, total = .0, .0
    unloader = transforms.ToPILImage()
    lalala = 0
    for batch_idx, (x1, x2, labels, x1_labels, x2_labels) in enumerate(validloader):

        x1, x2 = Variable(x1, requires_grad=True).float(), Variable(x2, requires_grad=True).float()
        labels = torch.tensor(labels)
        bs = x1.size(0)
        x1, x2, labels = x1.cuda(), x2.cuda(), labels.cuda()

        kin_prob, _ = kin_model(x1, x2)

        pred = (kin_prob > 0.5).long()
        results = (pred == (labels.long())).long()
        # print(kin_prob)

        ii = 0
        for op in kin_prob:
            if int(labels[ii]) == 1:
                evaluate_result.append('1 \t 0 \t ' + str(op.item()))
            if int(labels[ii]) == 0:
                evaluate_result.append('0 \t 1 \t ' + str(op.item()))
            ii += 1
    return evaluate_result


def save_model(tosave_model, epoch, k, best):
    model_path = str(args.relat) + '_C_' + str(args.batch_c) + '_P_' + str(
        args.part) + '_fold' + str(k) + '_bs' + str(args.batch_size) + '_' + str(best) + '.pth'

    save_path = os.path.join(args.save_model, model_path)
    torch.save(tosave_model.state_dict(), save_path)
    with open(os.path.join(args.save_model, 'checkpoint.txt'), 'w') as fin:
        fin.write(model_path + ' ' + str(epoch) + '\n')


def load_model(unload_model):
    if not os.path.exists(args.save_model):
        os.makedirs(args.save_model)
        print(args.save_model, 'is created!')
    if not os.path.exists(os.path.join(args.save_model, 'checkpoint.txt')):
        f = open(os.path.join(args.save_model, args.relat + '/checkpoint.txt'), 'w')
        print('checkpoint', 'is created!')

    start_index = 0
    with open(os.path.join(args.save_model, args.relat + '/checkpoint.txt'), 'r') as fin:
        lines = fin.readlines()
        if len(lines) > 0:
            model_path, model_index = lines[0].split()
            print('Resuming from', model_path)
            unload_model.load_state_dict(torch.load(os.path.join(args.save_model, model_path)))
            start_index = int(model_index) + 1
    return start_index


def build_model():
    kin_model = RelationNetwork()
    kin_model.cuda()
    torch.backends.cudnn.benchmark = True

    return kin_model


def self_binary_cross_entropy(x, y, reduction='mean'):
    loss = -((x + 1e-9).log() * y + (1 - x + 1e-9).log() * (1 - y))
    if reduction == 'mean':
        return loss.mean()
    else:
        return loss


courve_result = []
loss_result = []
family_result = []
for k in range(1, 6):

    kin_model = build_model()
    w_model = MetaNet().cuda()
    print('%s C %d T %d \nModel Built!' % (args.relat, args.batch_c, args.part))

    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_path = args.log + args.relat + '/' + time_str + '_C_' + str(args.batch_c) + '_P_' + str(
        args.part) + '_' + args.relat + "_k_fold_" + str(k) + "_lr_" + str(args.lr) + "_bs_" + str(
        args.batch_size) + "_" + ".txt"
    out_path = args.log + args.relat + '/_C_' + str(args.batch_c) + '_P_' + str(
        args.part) + '_' + args.relat + "_lr_" + str(args.lr) + "_bs_" + str(args.batch_size) + "_result_" + ".txt"
    # criterion = FocalLoss()
    criterion = torch.nn.BCELoss()
    criterion1 = torch.nn.CrossEntropyLoss()
    optimizer_kin = optim.Adam(params=kin_model.params(), lr=args.lr)

    optimizer_w = optim.Adam(params=w_model.params(), lr=args.meta_lr)
    # optimizer_w = torch.optim.SGD(w_model.params(), 1e-3,
    #                               momentum=0.9, nesterov=True,
    #                               weight_decay=5e-4)

    train_pos_set = train_pos_dataloader(relat=args.relat, k=k, data_root=args.data_root)
    train_pos_loader = torch.utils.data.DataLoader(train_pos_set, batch_size=args.batch_size // 2, shuffle=True,
                                                   num_workers=args.num_workers,
                                                   worker_init_fn=np.random.seed(args.manualSeed))

    train_neg_set = train_neg_dataloader(relat=args.relat, k=k, data_root=args.data_root, c=args.batch_c)
    train_neg_loader = torch.utils.data.DataLoader(train_neg_set, batch_size=args.batch_size // 2 * args.batch_c,
                                                   shuffle=True, num_workers=args.num_workers,
                                                   worker_init_fn=np.random.seed(args.manualSeed))

    train_meta_loader = meta_data_loader(batch_size=args.batch_size, relat=args.relat, k=k, data_root=args.data_root)

    validset = test_dataloader(relat=args.relat, k=k, data_root=args.data_root)
    validloader = torch.utils.data.DataLoader(validset, batch_size=args.valid_batch_size, shuffle=False,
                                              num_workers=args.num_workers,
                                              worker_init_fn=np.random.seed(args.manualSeed))
    print("Data loaded!")


    def train(epoch, k, log_path):
        kin_model.train()

        batch_time = aveMeter.AverageMeter()
        data_time = aveMeter.AverageMeter()
        losses = aveMeter.AverageMeter()
        familylosses = aveMeter.AverageMeter()
        acces = aveMeter.AverageMeter()

        end_time = time.time()

        log_file = open(log_path, 'a')
        log_file.write('Random seed:%d\n' % args.manualSeed)
        for batch_idx, (pos_sam, neg_sam) in enumerate(zip(train_pos_loader, train_neg_loader)):
            '''1. batch data prepare'''
            p_x1, p_x2, p_labels, pf1_labels, pf2_labels = pos_sam
            n_x1, n_x2, n_labels, nf1_labels, nf2_labels = neg_sam
            x1 = torch.cat((p_x1, n_x1), 0)  # 父p+n
            x2 = torch.cat((p_x2, n_x2), 0)  # 子p+n
            labels = torch.cat((p_labels, n_labels), 0)  # labels p+n
            family_labels_1 = torch.cat((pf1_labels, nf1_labels), dim=0)  # KI KII
            family_labels_2 = torch.cat((pf2_labels, nf2_labels), dim=0)
            family_labels = torch.cat((family_labels_1, family_labels_2), dim=0)

            bz = int(len(p_labels))

            x1, x2 = Variable(x1, requires_grad=False).float(), Variable(x2, requires_grad=False).float()
            labels = torch.tensor(labels)
            bs = x1.size(0)
            x1, x2, labels, family_labels = x1.cuda(), x2.cuda(), labels.cuda(), family_labels.cuda()

            data_time.update(time.time() - end_time)

            '''Actual Training the Kinship Model'''
            y_f, f_f = kin_model(x1, x2)
            cost_v = F.binary_cross_entropy(y_f, labels.float(), reduction='none')
            # cost_v = criterion(y_f, labels.float())
            loss6 = torch.tensor(0.0).cuda()
            num_part = args.part
            for i in range(num_part):
                loss6 += criterion1(f_f[i], family_labels.view(len(family_labels), ))

            familylosses.update(loss6.cpu().data.numpy() / num_part)

            l_f = torch.sum(cost_v)  # * w_lambda_norm.view(-1)
            losses.update(l_f.cpu().data.numpy())
            l_f += (loss6 / num_part)

            optimizer_kin.zero_grad()
            l_f.backward()
            optimizer_kin.step()
            # end training

            pred = (y_f > 0.5).long()
            results = (pred == (labels.long())).long()
            results = results.cpu().data.numpy()
            acc = sum(results) * 1.0 / len(results)
            acces.update(acc, bs)

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if batch_idx % args.print_freq == 0:
                # print(cost_v,w_lambda_norm.view(-1))
                print('K-Fold: [%d/5]  '
                      'Epoch: [%d][%d/%d]  '
                      'Time %.3f (%.3f)  '
                      'Data %.3f (%.3f)  '
                      'Loss %.3f (%.3f)  '
                      'Family Loss %.3f (%.3f) '
                      'Acc %.3f (%.3f)' % (k, epoch, batch_idx, len(train_pos_loader),
                                           batch_time.val, batch_time.avg, data_time.val, data_time.avg,
                                           losses.val, losses.avg, familylosses.val, familylosses.avg, acces.val,
                                           acces.avg))
                # print(www_p.data[0], www_n.data[0])
                log_file.write('K-Fold: [%d/5]  '
                               'Epoch: [%d][%d/%d]  '
                               'Time %.3f (%.3f)  '
                               'Data %.3f (%.3f)  '
                               'Loss %.3f (%.3f)  '
                               'Family Loss %.3f (%.3f) '
                               'Acc %.3f (%.3f)\n' % (k, epoch, batch_idx, len(train_pos_loader),
                                                      batch_time.val, batch_time.avg, data_time.val, data_time.avg,
                                                      losses.val, losses.avg, familylosses.val, familylosses.avg,
                                                      acces.val, acces.avg))
        log_file.close()
        loss_result.append(losses.avg)
        family_result.append(familylosses.avg)


    def train_l2rw(epoch, k, log_path):
        kin_model.train()

        batch_time = aveMeter.AverageMeter()
        data_time = aveMeter.AverageMeter()
        losses = aveMeter.AverageMeter()
        familylosses = aveMeter.AverageMeter()
        acces = aveMeter.AverageMeter()

        end_time = time.time()

        log_file = open(log_path, 'a')
        log_file.write('Random seed:%d\n' % args.manualSeed)
        for batch_idx, (pos_sam, neg_sam) in enumerate(zip(train_pos_loader, train_neg_loader)):
            '''1. batch data prepare'''
            p_x1, p_x2, p_labels, pf1_labels, pf2_labels = pos_sam
            n_x1, n_x2, n_labels, nf1_labels, nf2_labels = neg_sam
            x1 = torch.cat((p_x1, n_x1), 0)  # 父p+n
            x2 = torch.cat((p_x2, n_x2), 0)  # 子p+n
            labels = torch.cat((p_labels, n_labels), 0)  # labels p+n
            family_labels_1 = torch.cat((pf1_labels, nf1_labels), dim=0)  # KI KII
            family_labels_2 = torch.cat((pf2_labels, nf2_labels), dim=0)
            family_labels = torch.cat((family_labels_1, family_labels_2), dim=0)

            bz = int(len(p_labels))

            x1, x2 = Variable(x1, requires_grad=False).float(), Variable(x2, requires_grad=False).float()
            labels = torch.tensor(labels)
            bs = x1.size(0)
            x1, x2, labels, family_labels = x1.cuda(), x2.cuda(), labels.cuda(), family_labels.cuda()

            meta_x1, meta_x2, meta_labels = next(train_meta_loader)
            meta_x1, meta_x2 = Variable(meta_x1, requires_grad=False).float(), Variable(meta_x2,
                                                                                        requires_grad=False).float()
            meta_labels = torch.tensor(meta_labels)
            meta_x1, meta_x2, meta_labels = meta_x1.cuda(), meta_x2.cuda(), meta_labels.cuda()
            data_time.update(time.time() - end_time)

            meta_model = build_model()

            meta_model.load_state_dict(kin_model.state_dict())
            meta_model.train()
            # 1. Update meta model on training data
            meta_train_outputs, _ = meta_model(x1, x2)
            meta_train_loss = F.binary_cross_entropy(meta_train_outputs, labels.float(), reduction='none')
            eps = torch.zeros(meta_train_loss.size(), requires_grad=True, device='cuda:0')
            meta_train_loss = torch.sum(eps * meta_train_loss)

            meta_model.zero_grad()
            grads = torch.autograd.grad(meta_train_loss, (meta_model.params()), create_graph=True, allow_unused=True)
            meta_model.update_params(lr_inner=args.lr2, source_params=grads)

            # 2. Compute grads of eps on meta validation data

            meta_val_outputs, _ = meta_model(meta_x1, meta_x2)

            meta_val_loss = self_binary_cross_entropy(meta_val_outputs, meta_labels.float())
            eps_grads = torch.autograd.grad(meta_val_loss, eps)[0].detach()

            # 3. Compute weights for current training batch
            w_tilde = torch.clamp(-eps_grads, min=0)
            l1_norm = torch.sum(w_tilde)
            if l1_norm != 0:
                w = w_tilde / l1_norm
            else:
                w = w_tilde

            '''Pretraining the Kinship Model'''
            y_f, f_f = kin_model(x1, x2)
            cost_v = F.binary_cross_entropy(y_f, labels.float(), reduction='none')
            # cost_v = criterion(y_f, labels.float())
            loss6 = torch.tensor(0.0).cuda()
            num_part = args.part
            for i in range(num_part):
                loss6 += criterion1(f_f[i], family_labels.view(len(family_labels), ))

            familylosses.update(loss6.cpu().data.numpy() / num_part)

            l_f = torch.sum(w * cost_v)  # * w_lambda_norm.view(-1)
            losses.update(l_f.cpu().data.numpy())
            l_f += (loss6 / num_part)

            optimizer_kin.zero_grad()
            l_f.backward()
            optimizer_kin.step()
            # end training

            pred = (y_f > 0.5).long()
            results = (pred == (labels.long())).long()
            results = results.cpu().data.numpy()
            acc = sum(results) * 1.0 / len(results)
            acces.update(acc, bs)

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if batch_idx % args.print_freq == 0:
                # print(cost_v,w_lambda_norm.view(-1))
                print('K-Fold: [%d/5]  '
                      'Epoch: [%d][%d/%d]  '
                      'Time %.3f (%.3f)  '
                      'Data %.3f (%.3f)  '
                      'Loss %.3f (%.3f)  '
                      'Family Loss %.3f (%.3f) '
                      'Acc %.3f (%.3f)' % (k, epoch, batch_idx, len(train_pos_loader),
                                           batch_time.val, batch_time.avg, data_time.val, data_time.avg,
                                           losses.val, losses.avg, familylosses.val, familylosses.avg, acces.val,
                                           acces.avg))
                # print(www_p.data[0], www_n.data[0])
                log_file.write('K-Fold: [%d/5]  '
                               'Epoch: [%d][%d/%d]  '
                               'Time %.3f (%.3f)  '
                               'Data %.3f (%.3f)  '
                               'Loss %.3f (%.3f)  '
                               'Family Loss %.3f (%.3f) '
                               'Acc %.3f (%.3f)\n' % (k, epoch, batch_idx, len(train_pos_loader),
                                                      batch_time.val, batch_time.avg, data_time.val, data_time.avg,
                                                      losses.val, losses.avg, familylosses.val, familylosses.avg,
                                                      acces.val, acces.avg))
        log_file.close()
        loss_result.append(losses.avg)
        family_result.append(familylosses.avg)


    def train_epoch(epoch, k, log_path):
        kin_model.train()

        batch_time = aveMeter.AverageMeter()
        data_time = aveMeter.AverageMeter()
        losses = aveMeter.AverageMeter()
        familylosses = aveMeter.AverageMeter()
        acces = aveMeter.AverageMeter()
        metalosses = aveMeter.AverageMeter()
        metaacces = aveMeter.AverageMeter()

        end_time = time.time()

        log_file = open(log_path, 'a')
        log_file.write('Random seed:%d\n' % args.manualSeed)
        wp = []
        for batch_idx, (pos_sam, neg_sam) in enumerate(zip(train_pos_loader, train_neg_loader)):
            '''1. batch data prepare'''
            p_x1, p_x2, p_labels, pf1_labels, pf2_labels = pos_sam
            n_x1, n_x2, n_labels, nf1_labels, nf2_labels = neg_sam
            x1 = torch.cat((p_x1, n_x1), 0)  # 父p+n
            x2 = torch.cat((p_x2, n_x2), 0)  # 子p+n
            labels = torch.cat((p_labels, n_labels), 0)  # labels p+n
            family_labels_1 = torch.cat((pf1_labels, nf1_labels), dim=0)  # KI KII
            family_labels_2 = torch.cat((pf2_labels, nf2_labels), dim=0)
            family_labels = torch.cat((family_labels_1, family_labels_2), dim=0)

            bz = int(len(p_labels))

            x1, x2 = Variable(x1, requires_grad=False).float(), Variable(x2, requires_grad=False).float()
            labels = torch.tensor(labels)
            bs = x1.size(0)
            # x1, x2, labels = x1.cuda(), x2.cuda(), labels.cuda()
            x1, x2, labels, family_labels = x1.cuda(), x2.cuda(), labels.cuda(), family_labels.cuda()

            meta_x1, meta_x2, meta_labels = next(train_meta_loader)
            meta_x1, meta_x2 = Variable(meta_x1, requires_grad=False).float(), Variable(meta_x2,
                                                                                        requires_grad=False).float()
            meta_labels = torch.tensor(meta_labels)
            meta_x1, meta_x2, meta_labels = meta_x1.cuda(), meta_x2.cuda(), meta_labels.cuda()

            data_time.update(time.time() - end_time)

            '''2. One Step Gradient '''
            meta_model = build_model()
            meta_model.load_state_dict(kin_model.state_dict())
            meta_model.train()

            '''Virtual Training'''
            y_f_hat, _ = meta_model(x1, x2)
            cost_v = self_binary_cross_entropy(y_f_hat, labels.float(), reduction='none')
            w_lambda = w_model(cost_v.data)

            norm_c = torch.sum(w_lambda)

            if norm_c != 0:
                w_lambda_norm = w_lambda / norm_c
            else:
                w_lambda_norm = w_lambda

            l_f_meta = torch.sum(cost_v * w_lambda_norm.view(-1))
            meta_model.zero_grad()
            grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True, allow_unused=True)
            meta_model.update_params(lr_inner=args.lr2, source_params=grads)

            '''3. Update MetaNet with the one step gradient'''

            '''Training the Meta-Weight Network'''
            y_g_hat, _ = meta_model(meta_x1, meta_x2)
            l_g_meta = self_binary_cross_entropy(y_g_hat, meta_labels.float())
            metalosses.update(l_g_meta.cpu().data.numpy())

            pred = (y_g_hat > 0.5).long()
            results = (pred == (meta_labels.long())).long()
            results = results.cpu().data.numpy()
            acc = sum(results) * 1.0 / len(results)
            metaacces.update(acc)

            optimizer_w.zero_grad()
            l_g_meta.backward()
            optimizer_w.step()

            '''4. Update RN with MetaNet'''

            '''Actual Training'''
            y_f, f_f = kin_model(x1, x2)
            cost_v = F.binary_cross_entropy(y_f, labels.float(), reduction='none')

            loss6 = torch.tensor(0.0).cuda()
            num_part = args.part
            for i in range(num_part):
                loss6 += criterion1(f_f[i], family_labels.view(len(family_labels), ))

            familylosses.update(loss6.cpu().data.numpy() / num_part)
            with torch.no_grad():
                w_lambda = w_model(cost_v.data)
            norm_c = torch.sum(w_lambda)

            if norm_c != 0:
                w_lambda_norm = w_lambda / norm_c
            else:
                w_lambda_norm = w_lambda

            w_lambda_norm_p = w_lambda_norm[:bz]
            w_lambda_norm_n = w_lambda_norm[bz:]
            www_p = torch.sum(w_lambda_norm_p)
            www_n = torch.sum(w_lambda_norm_n)
            wp.append(www_p.item())
            l_f = torch.sum(cost_v * w_lambda_norm.view(-1))
            losses.update(l_f.cpu().data.numpy())
            l_f += (loss6 / num_part)

            optimizer_kin.zero_grad()
            l_f.backward()
            optimizer_kin.step()
            # end training

            pred = (y_f > 0.5).long()
            results = (pred == (labels.long())).long()
            results = results.cpu().data.numpy()
            acc = sum(results) * 1.0 / len(results)
            acces.update(acc, bs)

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if batch_idx % args.print_freq == 0:
                # print(cost_v,w_lambda_norm.view(-1))
                print('K-Fold: [%d/5]  '
                      'Epoch: [%d][%d/%d]  '
                      'Time %.3f (%.3f)  '
                      'Data %.3f (%.3f)  '
                      'MLoss %.3f (%.3f)  '
                      'MAcc %.3f (%.3f)  '
                      'Loss %.3f (%.3f)  '
                      'Family Loss %.3f (%.3f) '
                      'Acc %.3f (%.3f)' % (k, epoch, batch_idx, len(train_pos_loader),
                                           batch_time.val, batch_time.avg, data_time.val, data_time.avg,
                                           metalosses.val, metalosses.avg, metaacces.val, metaacces.avg,
                                           losses.val, losses.avg, familylosses.val, familylosses.avg, acces.val,
                                           acces.avg))
                # print(www_p.data[0], www_n.data[0])
                log_file.write('K-Fold: [%d/5]  '
                               'Epoch: [%d][%d/%d]  '
                               'Time %.3f (%.3f)  '
                               'Data %.3f (%.3f)  '
                               'Loss %.3f (%.3f)  '
                               'Family Loss %.3f (%.3f) '
                               'Acc %.3f (%.3f)\n' % (k, epoch, batch_idx, len(train_pos_loader),
                                                      batch_time.val, batch_time.avg, data_time.val, data_time.avg,
                                                      losses.val, losses.avg, familylosses.val, familylosses.avg,
                                                      acces.val, acces.avg))
        log_file.close()
        www = 0
        for ww in wp:
            www += ww
        www = www / len(wp)
        courve_result.append(str(www))
        loss_result.append(losses.avg)
        family_result.append(familylosses.avg)


    def valid_epoch(epoch, k, best_acc, log_path, best_epoch):
        kin_model.eval()

        batch_time = aveMeter.AverageMeter()
        data_time = aveMeter.AverageMeter()
        losses = aveMeter.AverageMeter()
        acces = aveMeter.AverageMeter()

        end_time = time.time()

        log_file = open(log_path, 'a')

        for batch_idx, (x1, x2, labels, x1_labels, x2_labels) in enumerate(validloader):
            data_time.update(time.time() - end_time)

            x1, x2 = Variable(x1, requires_grad=True).float(), Variable(x2, requires_grad=True).float()
            labels = torch.tensor(labels)
            bs = x1.size(0)
            x1, x2, labels = x1.cuda(), x2.cuda(), labels.cuda()

            kin_prob, _ = kin_model(x1, x2)

            loss = criterion(kin_prob, labels.float())

            losses.update(loss.cpu().data.numpy(), bs)

            pred = (kin_prob > 0.5).long()
            results = (pred == (labels.long())).long()
            results = results.cpu().data.numpy()
            acc = sum(results) * 1.0 / len(results)
            acces.update(acc, bs)

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if batch_idx % args.print_freq == 0:
                print('K-Fold: [%d/5]  '
                      'Valid Epoch: [%d][%d/%d]  '
                      'Time %.3f (%.3f)  '
                      'Data %.3f (%.3f)  '
                      'Loss %.3f (%.3f)  '
                      'Acc %.3f (%.3f)' % (k, epoch, batch_idx, len(validloader),
                                           batch_time.val, batch_time.avg, data_time.val, data_time.avg,
                                           losses.val, losses.avg, acces.val, acces.avg))
                log_file.write('K-Fold: [%d/5]  '
                               'Valid Epoch: [%d][%d/%d]  '
                               'Time %.3f (%.3f)  '
                               'Data %.3f (%.3f)  '
                               'Loss %.3f (%.3f)  '
                               'Acc %.3f (%.3f)\n' % (k, epoch, batch_idx, len(validloader),
                                                      batch_time.val, batch_time.avg, data_time.val, data_time.avg,
                                                      losses.val, losses.avg, acces.val, acces.avg))

        print("Valid: final acc: %.3f  "
              'Best Val Acc %.3f  '
              'Best Val Epoch: %d' % (acces.avg, best_acc, best_epoch))
        log_file.write("Valid: final acc: %.3f  "
                       "Best Val Acc %.3f  "
                       "Best Val Epoch: %d\n" % (acces.avg, best_acc, best_epoch))
        log_file.close()
        return acces.avg


    best_acc = -0.1
    best_epoch = 0
    for epoch_iter in range(1, args.max_epochs):
        lr_decay = args.lr_decay
        lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')] + [np.inf]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_kin, milestones=lr_decay_epoch, gamma=lr_decay,
                                                            last_epoch=-1)
        lr_scheduler.step()
        if epoch_iter < args.stop_epoch:
            train(epoch_iter, k, log_path)
        else:
            train_epoch(epoch_iter, k, log_path)
        with torch.no_grad():
            valid_acc = valid_epoch(epoch_iter, k, best_acc, log_path, best_epoch)

        log_file = open(log_path, 'a')

        if (best_acc < valid_acc):
            print("The best acc on Fold %d is getting better from %.3f to %.3f" % (k, best_acc, valid_acc))
            log_file.write("The best acc on Fold %d is getting better from %.3f to %.3f\n" % (k, best_acc, valid_acc))
            log_file.close()

            best_acc = valid_acc
            best_epoch = epoch_iter
            save_model(kin_model, epoch_iter, k, best_acc)
        if best_acc >= 0.999:
            break

    with open(out_path, 'a') as f:
        f.write("k_fold:  " + str(k) + '  ' + "Epoch %d Best Val Acc %.3f\n" % (best_epoch, best_acc))
