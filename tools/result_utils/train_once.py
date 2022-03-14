import sys
sys.path.insert(0, '../../')
import numpy as np
from torch import optim
from torch.utils.data import DataLoader

from src.dataset import SubsetSequentialSampler, collate_fn, SubsetRandomSampler
from src.dataset.cifar10 import Cifar10
from src.model import Learner
from tools.test_utils.test_utils import test
from tools.train_utils.train_utils import train


def txt_to_list(txt_path):
    ret = []
    with open(txt_path, 'r') as f:
        for line in f:
            ret.append(int(line))
    return ret

model_cfg = [('conv2d', [64,3,3,3,1,1]),('bn', [64]),('relu', [True]),
                ('conv2d', [64,64,3,3,1,1]),('bn', [64]),('relu', [True]),
                ('max_pool2d', [2,2,0]),
                ('conv2d', [128,64,3,3,1,1]),('bn', [128]),('relu', [True]),
                ('conv2d', [128,128,3,3,1,1]),('bn', [128]),('relu', [True]),
                ('max_pool2d', [2,2,0]),
                ('conv2d', [256,128,3,3,1,1]),('bn', [256]),('relu', [True]),
                ('conv2d', [256,256,3,3,1,1]),('bn', [256]),('relu', [True]),
                ('conv2d', [256,256,3,3,1,1]),('bn', [256]),('relu', [True]),
                ('max_pool2d', [2,2,0]),
                ('conv2d', [512,256,3,3,1,1]),('bn', [512]),('relu', [True]),
                ('conv2d', [512,512,3,3,1,1]),('bn', [512]),('relu', [True]),
                ('conv2d', [512,512,3,3,1,1]),('bn', [512]),('relu', [True]),
                ('max_pool2d', [2,2,0]),
                ('conv2d', [512,512,3,3,1,1]),('bn', [512]),('relu', [True]),
                ('conv2d', [512,512,3,3,1,1]),('bn', [512]),('relu', [True]),
                ('conv2d', [512,512,3,3,1,1]),('bn', [512]),('relu', [True]),
                ('max_pool2d', [2,2,0]),
                ('flatten', []),
                ('linear', [512, 512]),
                ('relu', [True]),
                ('linear', [512, 512]),
                ('relu', [True]),
                ('linear', [10, 512])]


if __name__ == '__main__':
    train_dataset = Cifar10(split='train')
    txt_path = '/cpfs2/user/wangjie/PythonProjects/Ml_Model/output/epoch_5_meta/result/meta_al_entropy/labeled_index_10.txt'
    labeled_index = txt_to_list(txt_path)
    acc_list = []
    labeled_dataloader = DataLoader(train_dataset,
                                    batch_size=4,
                                    sampler=SubsetSequentialSampler(labeled_index),
                                    collate_fn=collate_fn)
    test_dataset = Cifar10(split='test')
    test_dataloader = DataLoader(test_dataset,
                                batch_size=4,
                                collate_fn=collate_fn)
    model = Learner(model_cfg)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model = train(None, model, optimizer, labeled_dataloader)
    acc = test(model, test_dataloader,None)
    acc_list.append(acc)
    print(acc_list)
