import torch
from torch import optim
from torch.nn import functional as F

from src.model import Learner

device = torch.device('cuda')


def test(model, test_dataloater, tb_log):
    total_correct = 0
    totol_nums = 0
    for step, (input, label) in enumerate(test_dataloater):
        input, label = input.to(device), label.to(device)
        logits = model(input, vars=None, bn_training=False)
        pred = F.softmax(logits, dim=1).argmax(dim=1)
        correct = torch.eq(pred, label).sum().item()
        total_correct += correct
        totol_nums += input.size(0)

    acc = total_correct / totol_nums
    return acc


def test_classes(model, test_dataloader, tb_log):
    """
    :return: 每个类别的准确率
    """
    correct = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for step, (input, label) in enumerate(test_dataloader):
        input, label = input.to(device), label.to(device)
        logits = model(input, vars=None, bn_training=False)
        pred = F.softmax(logits, dim=1).argmax(dim=1)
        for i in range(len(input[0])):
            if (pred[i] == label[i]):
                correct[pred[i]] += 1
    acc_list = [x / 1000 for x in correct]
    return acc_list, sum(acc_list)/10
