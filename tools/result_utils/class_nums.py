import numpy as np

from src.dataset.cifar10 import Cifar10
import matplotlib.pyplot as plt


def txt_to_list(txt_path):
    ret = []
    with open(txt_path, 'r') as f:
        for line in f:
            ret.append(int(line))
    return ret


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.2, 1.03 * height, '%s' % int(height))


if __name__ == '__main__':
    '''
    txt_path = 'output0/result/labeled_index_{}.txt'
    cifar10 = Cifar10(split='train')
    name = ['apl', 'atmb', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for i in range(1, 2):
        path = txt_path.format(i)
        ret = txt_to_list(path)
        val = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for index in ret:
            label = cifar10[index][1]
            val[label] += 1
        autolabel(plt.bar(range(len(val)), val, tick_label=name))
        plt.show()
    
    txt_path = 'output0/result/labeled_index_1.txt'
    labeled_index = txt_to_list(txt_path)[:1000]
    random_list = []
    for i in range(40):
        unlabeled_index = [x for x in list(range(50000)) if x not in labeled_index]
        print(len(unlabeled_index))
        picked_index = np.random.choice(unlabeled_index, 500, replace=False).tolist()
        labeled_index.extend(picked_index)
        random_list.append(labeled_index.copy())
    cifar10 = Cifar10(split='train')
    name = ['apl', 'atmb', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(0, 40, 4):
        val = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for index in random_list[i]:
            label = cifar10[index][1]
            val[label] += 1
        autolabel(plt.bar(range(len(val)), val, tick_label=name))
        plt.show()
    '''
    # 每个类增长率变化
    cifar10 = Cifar10(split='train')
    pickled_label_dir = "../../output0/result/picked_index_{}.txt"
    name = ['apl', 'atmb', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_nums = [[0]*40, [0]*40, [0]*40, [0]*40, [0]*40, [0]*40, [0]*40, [0]*40, [0]*40, [0]*40]
    cls_list = [0]*10
    labeled_index = []
    all_index = list(range(50000))
    for i in range(1, 41):
        # al样本
        # picked_label_path = pickled_label_dir.format(str(i))
        # index_list = txt_to_list(picked_label_path)
        # 随机样本
        index_list = np.random.choice(all_index, 500, replace=False).tolist()
        labeled_index.extend(index_list)
        all_index = [x for x in all_index if x not in labeled_index]
        print(len(all_index))
        for index in index_list:
            label = cifar10[index][1]
            class_nums[label][i-1] += 1
            cls_list[label] += 1
    x = list(range(1, 41))
    plt.bar(x, class_nums[0], label=name[0])
    bottom = class_nums[0]
    for i in range(1, 10):
        plt.bar(x, class_nums[i], bottom=bottom, label=name[i])
        bottom = np.sum([bottom, class_nums[i]], axis=0).tolist()
    plt.legend()
    plt.show()
    print(cls_list)