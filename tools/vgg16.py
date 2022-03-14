import _init_path
import os
import ssl
import torch
import torchvision
import torchvision.transforms as transforms
import math
import torch
import torch.nn as nn
from src.model import Learner
import numpy as np
from torch.nn import functional as F

if __name__ == '__main__':

    ssl._create_default_https_context = ssl._create_unverified_context
    ########################################
    #第1步：载入数据
    ########################################

    #使用torchvision可以很方便地下载cifar10数据集，而torchvision下载的数据集为[0, 1]的PILImage格式，我们需要将张量Tensor归一化到[-1, 1]

    transform = transforms.Compose(
        [transforms.ToTensor(), #将PILImage转换为张量
         transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],std=[0.2675, 0.2565, 0.2761])] #将[0, 1]归一化到[-1, 1]
         )

    from src.dataset import collate_fn, SubsetSequentialSampler
    l_index = np.random.choice(list(range(50000)), 21000, replace = False)
    trainset = torchvision.datasets.CIFAR10(root='../classifier_cifar10/data', #root表示cifar10的数据存放目录，使用torchvision可直接下载cifar10数据集，也可直接在https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz这里下载（链接来自cifar10官网）
                                            train=True,
                                            download=True,
                                            transform=transform #按照上面定义的transform格式转换下载的数据
                                            )
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=4, #每个batch载入的图片数量，默认为1
                                              sampler=SubsetSequentialSampler(l_index),
                                              num_workers=2 #载入训练数据所需的子任务数
                                              )

    testset = torchvision.datasets.CIFAR10(root='../classifier_cifar10/data',
                                           train=False,
                                           download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=2)

    cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #
    # ########################################
    # #查看训练数据
    # #备注：该部分代码可以不放入主函数
    # ########################################
    import numpy as np

    dataiter = iter(trainloader) #随机从训练数据中取一些数据
    print(trainloader)
    images, labels = dataiter.next()
    images.shape #(4L, 3L, 32L, 32L)
    #我们可以看到images的shape是4*3*32*32，原因是上面载入训练数据trainloader时一个batch里面有4张图片

########################################
# 第2步：构建卷积神经网络
########################################

    cfg = {'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}

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

    
    net = Learner(model_cfg)

    ########################################
    # 第3步：定义损失函数和优化方法
    ########################################
    import torch.optim as optim

    # x = torch.randn(2,3,32,32)
    # y = net(x)
    # print(y.size())
    criterion = nn.CrossEntropyLoss()  # 定义损失函数：交叉熵
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 定义优化方法：随机梯度下降

    ########################################
    # 第4步：卷积神经网络的训练
    ########################################
    for epoch in range(5):  # 训练数据集的迭代次数，这里cifar10数据集将迭代2次
        train_loss = 0.0
        for step, (x, y) in enumerate(trainloader):
            logits = net(x, vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 查看网络训练状态
            train_loss += loss.item()
            if step % 2000 == 1999:  # 每迭代2000个batch打印看一次当前网络收敛情况
                print('[%d, %5d] loss: %.3f' % (epoch + 1, step + 1, train_loss / 2000))
                train_loss = 0.0

        print('Saving epoch %d model ...' % (epoch + 1))
        state = {
            'net': net.state_dict(),
            'epoch': epoch + 1,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/cifar10_epoch_%d.ckpt' % (epoch + 1))

    print('Finished Training')
    ########################################
    # 第5步：批量计算整个测试集预测效果
    ########################################
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 当标记的label种类和预测的种类一致时认为正确，并计数
    '''
    total_correct = 0
    totol_nums = 0
    with torch.no_grad():
        for step, (input, label) in enumerate(testloader):
            logits = net(input, vars=None, bn_training=False)
            pred = F.softmax(logits, dim=1).argmax(dim=1)
            correct = torch.eq(pred, label).sum().item()
            total_correct += correct
            totol_nums += input.size(0)
    
    '''
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    # 结果打印：Accuracy of the network on the 10000 test images: 73 %

    ########################################
    # 分别查看每个类的预测效果
    ########################################
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            cifar10_classes[i], 100 * class_correct[i] / class_total[i]))


