import os
import pickle

import cv2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms

from src.dataset import collate_fn


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


class Cifar10(Dataset):
    def __init__(self, split=None):
        super(Cifar10, self).__init__()
        # self.root_path = cfg['ROOT_PATH']
        self.split = split
        self.data = []
        self.label = []
        if self.split == 'train':
            for i in range(1, 6):
                pickle = unpickle("/cpfs2/user/wangjie/PythonProjects/Ml_Model/data/cifar-10/data_batch_" + str(i))
                data, label = pickle['data'], pickle['labels']
                self.data.append(data)
                self.label.extend(label)
        elif self.split == 'test':
            pickle = unpickle("/cpfs2/user/wangjie/PythonProjects/Ml_Model/data/cifar-10/test_batch")
            data, label = pickle['data'], pickle['labels']
            self.data.append(data)
            self.label.extend(label)
        else:
            print("split is not true")
            return RuntimeError

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.label = np.array(self.label)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761])
        ])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]

        img = Image.fromarray(img)
        return self.transform(img), label


class MetaCifar10(Cifar10):
    """
    每次获取的是一个小任务的S和Q，以及对应的标签
    这个数据集包含整个train数据集，需要将其划分为labeled 数据集和 unlabeled数据集， 训练集包含50000个样本
    元学习使用的是labeled数据集，active learning使用的是unlabeled数据。从labeled中采样S和Q
    每个子任务每个类对应的标签应该和原始数据集一样
    """

    def __init__(self, n_task, n_support, n_query, split=None):
        super(MetaCifar10, self).__init__(split)
        self.nums_train_data = super(MetaCifar10, self).__len__()
        self.n_task = n_task
        self.n_suport = n_support
        self.n_query = n_query
        self.labeled_index = None

    def get_labeled_index(self):
        labeled_index = self.labeled_index
        return labeled_index

    def get_unlabeled_index(self):
        unlabeled_index = [x for x in range(len(self.label)) if x not in self.labeled_index]
        return unlabeled_index

    def set_labeled_index(self, labeled_index):
        self.labeled_index = labeled_index

    def __len__(self):
        # 每个epoch我们要抽取的小任务的总数
        return self.n_task

    def __getitem__(self, index):
        # 每次获取的是一个任务的支持集S和查询集Q， 随机采样，和index没有关系
        support, query = [], []
        # 这里S和Q是不相交的，或许可以改为相交
        idx_list = np.random.choice(self.labeled_index, self.n_suport + self.n_query)
        support_idx, query_idx = idx_list[:self.n_suport], idx_list[self.n_suport:]
        for idx in support_idx:
            support.append(self.transform(Image.fromarray(self.data[idx])))
        for idx in query_idx:
            query.append(self.transform(Image.fromarray(self.data[idx])))
        support = torch.stack(support)
        query = torch.stack(query)
        support_labels = self.label[support_idx]
        query_labels = self.label[query_idx]

        return support, query, support_labels, query_labels


if __name__ == '__main__':
    train_set = MetaCifar10(None)
    train_loader = DataLoader(
        train_set, 4,
        collate_fn=collate_fn, num_workers=0, pin_memory=True)  # 多线程会报错，可能因为多个task随机采样，会访问相同的内存
    it = iter(train_loader)
    data = next(it)
