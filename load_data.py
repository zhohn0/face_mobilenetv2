import numpy as np
import os
import torch
import torchvision
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import shutil

def calc_mean_std(dataset, axis=0):
    return (np.mean(dataset, axis=axis) / 255.0).tolist(), (np.std(dataset, axis=axis) / 255.0).tolist()


def get_cifar10_dataset(batch_size = 24):
    # 初始化的参数
    num_workers = 2
    input_size = 32
    data_path = './data/cifar10'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # 获取初始的均值和方差
    mean, std = calc_mean_std(torchvision.datasets.CIFAR10(root=data_path, train=True, download=True).train_data, axis=(0, 1, 2))
    # print(mean, std)
    transform_train = transforms.Compose([
                                            #transforms.Resize(32),
                                            transforms.RandomCrop(input_size, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ColorJitter(0.3, 0.3, 0.3),
                                            transforms.ToTensor(),
                                            #transforms.Normalize(mean, std)
                                          ])
    # 测试集是不是也应该进行resize
    transform_test = transforms.Compose([transforms.Resize(input_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])

    train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,  shuffle=True, num_workers=num_workers)

    test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

# 将cifar10保存成可见图片
def save_cifar10_image():
    index = 1
    batch_size = 24
    train_loader, test_loader = get_cifar10_dataset()
    for im, label in train_loader:
        for b in range(batch_size):
            img = im[b].data.permute(1, 2, 0)
            #img = im[b].numpy().transpose(1, 2, 0)
            plt.imshow(img)
            plt.imsave('./data/cifar10_all/cifar_{}.jpg'.format(index), img)
            index += 1

# 根据现有图片分成训练集和测试集
def split_face():
    count =0
    input_dir = '.\\data\\face_classifier\\train\\others'
    output_dir = '.\\data\\face_classifier\\test\\others'
    # 新建输出文件夹
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for (path, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            index = filename.split('.')[0].split('_')[1]
            index = int(index)#将x转换为一个整数
            if index % 10 == 0:
                src = path + '/' + filename
                dst = output_dir
                shutil.move(src, dst)
                count += 1
    print(count)

# 人脸的二分类数据集
def get_face_classifier_dataset(batch_size=128):
    # 初始化参数
    train_path = './data/face_classifier/train'
    test_path = './data/face_classifier/test'
    transform = transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor()])
    train_data = torchvision.datasets.ImageFolder(train_path, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(test_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader



if __name__ == '__main__':
    #train_loader, test_loader = get_cifar10_dataset()
    train_loader, test_loader = get_face_classifier_dataset()
    #save_cifar10_image()
    #split_face()
    for im, label in test_loader:
        print(label.shape)


