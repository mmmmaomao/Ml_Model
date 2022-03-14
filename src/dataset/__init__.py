import torch
from torch.utils.data import Sampler
from random import random


def SubsetRandomSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        random.shuffle(self.indices)
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


# 将多个task的数据拼接到一起
def meta_collate_fn(batch):
    support, query, support_label, query_label = [], [], [], []
    for s, q, sl, ql in batch:
        support.append(s)
        query.append(q)
        support_label.append(torch.from_numpy(sl))
        query_label.append(torch.from_numpy(ql))

    support = torch.stack(support)  # [n_ep, n_support, C, H, W]
    query = torch.stack(query)  # [n_ep, n_query, C, H, W]
    support_label = torch.stack(support_label)  # [n_ep, n_support]
    query_label = torch.stack(query_label)  # [n_ep, n_query]
    return support, query, support_label, query_label


def collate_fn(batch):
    input, label = [], []
    for x, y in batch:
        input.append(x)
        label.append(y)
    input = torch.stack(input)
    label = torch.LongTensor(label)
    return input, label


def build_dataloader():
    pass
