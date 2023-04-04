import os
from random import sample

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

transform_train = transforms.Compose([
    transforms.Resize(73),
    transforms.RandomCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(73),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def read_img(path, train_flag=True):
    img = Image.open(path)
    if train_flag == True:
        img = transform_train(img)
    else:
        img = transform_test(img)
    return img


class test_dataloader(data.Dataset):
    def __init__(self, relat="fs", k=1, data_root='/home/'):
        super(test_dataloader, self).__init__()
        self.isTrain = False
        self.k = k
        self.data_root = data_root
        self.images_root = self.data_root + "images/" + relat + '/'
        fold_root = self.data_root + "labels/" + relat + ".csv"
        csv_data = pd.read_csv(fold_root)
        csv_data = csv_data[csv_data['fold'] == k]

        csv_data = {'fold': pd.Series(csv_data['fold'].values), 'label': pd.Series(csv_data['label'].values),
                    'p1': pd.Series(csv_data['p1'].values), 'p2': pd.Series(csv_data['p2'].values)}

        self.data = pd.DataFrame(csv_data)

    def __getitem__(self, i):
        img1_path = os.path.join(self.images_root + self.data.loc[i]['p1'])
        img2_path = os.path.join(self.images_root + self.data.loc[i]['p2'])

        img1 = read_img(img1_path, self.isTrain)
        img2 = read_img(img2_path, self.isTrain)

        label = self.data.loc[i]['label']
        image1_label = torch.LongTensor([int(self.data.loc[i]['p1'].split('_')[1]) - 1])
        image2_label = torch.LongTensor([int(self.data.loc[i]['p2'].split('_')[1]) - 1])

        return img1, img2, label, image1_label, image2_label

    def __len__(self):
        return len(self.data)


def meta_data_loader(batch_size, relat="fs", k=1, data_root='/home/'):
    images_root = data_root + "images/" + relat + '/'
    fold_root = data_root + "labels/" + relat + ".csv"
    csv_data = pd.read_csv(fold_root)
    csv_data = csv_data[csv_data['fold'] != k]
    csv_data = csv_data[csv_data['label'] == 1]

    csv_data = {'fold': pd.Series(csv_data['fold'].values), 'label': pd.Series(csv_data['label'].values),
                'p1': pd.Series(csv_data['p1'].values), 'p2': pd.Series(csv_data['p2'].values)}

    data = pd.DataFrame(csv_data)
    data_size = len(data)
    images1 = []
    images2 = []
    for i in range(len(data)):
        img1_path = os.path.join(images_root + data.loc[i]['p1'])
        img2_path = os.path.join(images_root + data.loc[i]['p2'])
        images1.append(img1_path)
        images2.append(img2_path)

    while True:
        pos_sample = sample(range(data_size), batch_size // 2)

        neg_sample = [sample(range(data_size), 2) for i in range(batch_size // 2)]
        neg_sample = np.array(neg_sample)
        batch_pos_images1 = [read_img(images1[i], True) for i in pos_sample]
        batch_pos_images1 = torch.stack(batch_pos_images1, dim=0)
        batch_pos_images2 = [read_img(images2[i], True) for i in pos_sample]
        batch_pos_images2 = torch.stack(batch_pos_images2, dim=0)

        batch_neg_images1 = [read_img(images1[i], True) for i in neg_sample[:, 0]]
        batch_neg_images1 = torch.stack(batch_neg_images1, dim=0)
        batch_neg_images2 = [read_img(images2[i], True) for i in neg_sample[:, 1]]
        batch_neg_images2 = torch.stack(batch_neg_images2, dim=0)

        labels = [1 if i < batch_size // 2 else 0 for i in range((batch_size // 2) * 2)]

        yield torch.cat((batch_pos_images1, batch_neg_images1), dim=0), torch.cat(
            (batch_pos_images2, batch_neg_images2), dim=0), torch.tensor(labels)


class train_pos_dataloader(data.Dataset):
    def __init__(self, relat="fs", k=1, data_root='/home/'):
        super(train_pos_dataloader, self).__init__()
        self.isTrain = True
        self.k = k
        self.data_root = data_root
        self.images_root = self.data_root + "images/" + relat + '/'
        fold_root = self.data_root + "labels/" + relat + ".csv"
        csv_data = pd.read_csv(fold_root)
        csv_data = csv_data[csv_data['fold'] != k]
        csv_data = csv_data[csv_data['label'] == 1]

        csv_data = {'fold': pd.Series(csv_data['fold'].values), 'label': pd.Series(csv_data['label'].values),
                    'p1': pd.Series(csv_data['p1'].values), 'p2': pd.Series(csv_data['p2'].values)}

        self.data = pd.DataFrame(csv_data)

    def __getitem__(self, i):
        img1_path = os.path.join(self.images_root + self.data.loc[i]['p1'])
        img2_path = os.path.join(self.images_root + self.data.loc[i]['p2'])

        img1 = read_img(img1_path, self.isTrain)
        img2 = read_img(img2_path, self.isTrain)

        image1_label = torch.LongTensor([int(self.data.loc[i]['p1'].split('_')[1]) - 1])
        image2_label = torch.LongTensor([int(self.data.loc[i]['p2'].split('_')[1]) - 1])

        return img1, img2, torch.tensor(1, dtype=torch.int), image1_label, image2_label

    def __len__(self):
        return len(self.data)


class train_neg_dataloader(data.Dataset):
    def __init__(self, relat="fs", k=1, data_root='/home/', c=2):
        super(train_neg_dataloader, self).__init__()
        self.isTrain = True
        self.k = k
        self.data_root = data_root
        self.images_root = self.data_root + "images/" + relat + '/'
        fold_root = self.data_root + "labels/" + relat + ".csv"
        csv_data = pd.read_csv(fold_root)
        csv_data = csv_data[csv_data['fold'] != k]
        csv_data = csv_data[csv_data['label'] == 1]

        csv_data = {'fold': pd.Series(csv_data['fold'].values), 'label': pd.Series(csv_data['label'].values),
                    'p1': pd.Series(csv_data['p1'].values), 'p2': pd.Series(csv_data['p2'].values)}

        self.data = pd.DataFrame(csv_data)
        self.data_len = len(self.data) * c

    def __getitem__(self, i):
        neg_sample = sample(range(len(self.data)), 2)

        img1_path = os.path.join(self.images_root + self.data.loc[neg_sample[0]]['p1'])
        img2_path = os.path.join(self.images_root + self.data.loc[neg_sample[1]]['p2'])

        img1 = read_img(img1_path, self.isTrain)
        img2 = read_img(img2_path, self.isTrain)

        image1_label = torch.LongTensor([int(self.data.loc[neg_sample[0]]['p1'].split('_')[1]) - 1])
        image2_label = torch.LongTensor([int(self.data.loc[neg_sample[1]]['p2'].split('_')[1]) - 1])

        return img1, img2, torch.tensor(0, dtype=torch.int), image1_label, image2_label

    def __len__(self):
        return self.data_len
