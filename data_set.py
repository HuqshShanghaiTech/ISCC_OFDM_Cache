import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import random
import math

def gen_txt(dir, train):
    folder_level_list = os.listdir(dir)
    folder_level_list.sort()
    for folder_level in folder_level_list:
        for folder_label in os.listdir(dir + folder_level):
            for file in os.listdir(os.path.join(dir, folder_level, folder_label)):
                name = os.path.join(dir, folder_level, folder_label, file) + ' ' + str(int(folder_label) - 1) + '\n'
                train.write(name)
    train.close()


# def gen_txt_origin(dir, train, test):
#     folder_label_list = os.listdir(dir)
#     folder_label_list.sort()
#     for folder_label in folder_label_list[0:5]:
#         file_list = os.listdir(os.path.join(dir, folder_label))
#         file_list.sort()
#         label_idx = int(folder_label[-1])-1
#         for idx in range(0, 130):
#             name = os.path.join(dir, folder_label, file_list[idx * 2]) + ' ' + str(label_idx) + '\n'
#             train.write(name)
#         for idx in range(130, 230):
#             name = os.path.join(dir, folder_label, file_list[idx * 2]) + ' ' + str(label_idx) + '\n'
#             test.write(name)
#     train.close()
#     test.close()

def gen_txt_origin(dir, train, test, num_class):
    folder_label_list = os.listdir(dir)
    folder_label_list.sort()
    for folder_label in folder_label_list[0:num_class]:
        file_list = os.listdir(os.path.join(dir, folder_label))
        file_list.sort()
        label_idx = int(folder_label[-1]) - 1
        for idx in range(0, 600):
            name = os.path.join(dir, folder_label, file_list[idx]) + ' ' + str(label_idx) + '\n'
            train.write(name)
        for idx in range(1200, 1800):
            name = os.path.join(dir, folder_label, file_list[idx]) + ' ' + str(label_idx) + '\n'
            test.write(name)
    train.close()
    test.close()


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform, loader=default_loader):
        super(MyDataset, self).__init__()
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            words = line.split()
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):
        fn, label = self.imgs[item]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

class MyDatasetrain_1(Dataset):
    def __init__(self, txt, transform, num, num_hat, tmp_round, loader=default_loader):
        global content_old_1

        super(MyDatasetrain_1, self).__init__()

        fh = open(txt, 'r')
        contents = fh.readlines()
        imgs = []
        for i in range(7):
            #content = contents[i * 600: (i + 1) * 600]
            content = contents
            num = int(math.floor(num))
            num_hat = int(math.floor(num_hat))

            content = random.sample(content, num)
            if tmp_round == 1:
                content_train = content
            else:
                if num_hat == 0:
                    content_train = content
                else:
                    #print(len(content_old_1))
                    #print('num_hat', num_hat)
                    #content_cache = random.sample(content_old_1, num_hat)
                    content_cache = random.sample(contents, num_hat)
                    content_train = content + content_cache
            # print(content)
            imgs = imgs + [[i.split(' ')[0], int(i.split(' ')[1])] for i in content_train]
            # print(imgs)

            content_old_1 = content

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.num = num

    def __getitem__(self, item):
        fn, label = self.imgs[item]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

class MyDatasetrain_2(Dataset):
    def __init__(self, txt, transform, num, num_hat, tmp_round, loader=default_loader):


        super(MyDatasetrain_2, self).__init__()

        fh = open(txt, 'r')
        contents = fh.readlines()
        global content_old_2
        imgs = []
        for i in range(7):
            #content = contents[i * 600: (i + 1) * 600]
            content = contents
            num = int(math.floor(num))
            num_hat = int(math.floor(num_hat))

            content = random.sample(content, num)
            if tmp_round == 1:
                content_train = content
            else:
                if num_hat == 0:
                    content_train = content
                else:

                    #content_cache = random.sample(content_old_2, num_hat)
                    content_cache = random.sample(contents, num_hat)
                    content_train = content + content_cache
            # print(content)
            imgs = imgs + [[i.split(' ')[0], int(i.split(' ')[1])] for i in content_train]
            # print(imgs)
            content_old_2 = content
        #print('length', len(content_old_2))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.num = num

    def __getitem__(self, item):
        fn, label = self.imgs[item]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

class MyDatasetrain_3(Dataset):
    def __init__(self, txt, transform, num, num_hat, tmp_round, loader=default_loader):
        global content_old_3

        super(MyDatasetrain_3, self).__init__()

        fh = open(txt, 'r')
        contents = fh.readlines()

        imgs = []
        for i in range(7):
            #content = contents[i * 600: (i + 1) * 600]
            content = contents
            num = int(math.floor(num))
            num_hat = int(math.floor(num_hat))

            content = random.sample(content, num)
            if tmp_round == 1:
                content_train = content
            else:
                if num_hat == 0:
                    content_train = content
                else:

                    content_cache = random.sample(contents, num_hat)
                    #content_cache = random.sample(content_old_3, num_hat)
                    content_train = content + content_cache
            # print(content)
            imgs = imgs + [[i.split(' ')[0], int(i.split(' ')[1])] for i in content_train]
            # print(imgs)

            content_old_3 = content

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.num = num

    def __getitem__(self, item):
        fn, label = self.imgs[item]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

class MyDatasetrain_4(Dataset):
    def __init__(self, txt, transform, num, num_hat, tmp_round, loader=default_loader):
        global content_old_4

        super(MyDatasetrain_4, self).__init__()

        fh = open(txt, 'r')
        contents = fh.readlines()

        imgs = []
        for i in range(7):
            #content = contents[i * 600: (i + 1) * 600]
            content = contents
            num = int(math.floor(num))
            num_hat = int(math.floor(num_hat))

            content = random.sample(content, num)
            if tmp_round == 1:
                content_train = content
            else:
                if num_hat == 0:
                    content_train = content
                else:

                    #content_cache = random.sample(content_old_4, num_hat)
                    #print(len(content))
                    content_cache = random.sample(contents, num_hat)
                    content_train = content + content_cache
            # print(content)
            imgs = imgs + [[i.split(' ')[0], int(i.split(' ')[1])] for i in content_train]
            # print(imgs)

            content_old_4 = content

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.num = num

    def __getitem__(self, item):
        fn, label = self.imgs[item]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

class MyDatasetrain_5(Dataset):
    def __init__(self, txt, transform, num, num_hat, tmp_round, loader=default_loader):
        global content_old_5

        super(MyDatasetrain_5, self).__init__()

        fh = open(txt, 'r')
        contents = fh.readlines()

        imgs = []
        for i in range(7):
            content = contents[i * 600: (i + 1) * 600]
            content = contents
            num = int(math.floor(num))
            num_hat = int(math.floor(num_hat))

            content = random.sample(content, num)

            if tmp_round == 1:
                content_train = content
            else:
                if num_hat == 0:
                    content_train = content
                else:
                    #content_cache = random.sample(content_old_5, num_hat)
                    content_cache = random.sample(contents, num_hat)
                    content_train = content + content_cache
            # print(content)
            imgs = imgs + [[i.split(' ')[0], int(i.split(' ')[1])] for i in content_train]
            # print(imgs)

            content_old_5 = content

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.num = num

    def __getitem__(self, item):
        fn, label = self.imgs[item]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

class MyDatasetrain_6(Dataset):
    def __init__(self, txt, transform, num, num_hat, tmp_round, loader=default_loader):
        global content_old_6

        super(MyDatasetrain_6, self).__init__()

        fh = open(txt, 'r')
        contents = fh.readlines()

        imgs = []
        for i in range(7):
            content = contents[i * 600: (i + 1) * 600]
            content = contents
            num = int(math.floor(num))
            num_hat = int(math.floor(num_hat))

            content = random.sample(content, num)
            if tmp_round == 1:
                content_train = content
            else:
                if num_hat == 0:
                    content_train = content
                else:

                    #content_cache = random.sample(content_old_6, num_hat)
                    content_cache = random.sample(contents, num_hat)
                    content_train = content + content_cache
            # print(content)
            imgs = imgs + [[i.split(' ')[0], int(i.split(' ')[1])] for i in content_train]
            # print(imgs)

            content_old_6 = content

        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.num = num

    def __getitem__(self, item):
        fn, label = self.imgs[item]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class CenDataset(Dataset):
    def __init__(self, data, labels):
        super(CenDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, item):
        img = self.data[item]
        label = self.labels[item]
        return img, label

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    gen_txt_flag = True
    if gen_txt_flag:
        # # Data generated at the beginning and 20% accuracy
        # dir_train = './data/spect/radar_1/fig_train/'
        # train = open('./data/spect/radar_1/train.txt', 'w')
        # gen_txt(dir_train, train)
        #
        # dir_test = './data/spect/radar_1/fig_test/'
        # test = open('./data/spect/radar_1/test.txt', 'w')
        # gen_txt(dir_test, test)

        dir = './data/spect/THREE_RADAR_3000/radar_3/'
        train_1 = open(dir + 'train_1_m7.txt', 'w')
        train_2 = open(dir + 'test_m7.txt', 'w')
        num_class = 7
        gen_txt_origin(dir, train_1, train_2, num_class)
