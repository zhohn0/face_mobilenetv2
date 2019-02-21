# TODO
# 可以试着继续训练，加载已经保存的模型参数继续训练
# 2018/12/24 已经保存人脸二分类问题的模型

import os
import torch
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from mobilenet import MobileNetV2
from load_data import get_cifar10_dataset, get_face_classifier_dataset

num_epoches = 20
base_lr = 0.0001
lr_decay = 0.98

# 获取网络
net = MobileNetV2(n_class=2, input_size=96)
if torch.cuda.is_available():
    net = net.cuda()
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化方法
optimizer = optim.RMSprop(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4)

# 每epoch修改学习率
def adjust_lr(optimiter, epoch):
    learning_rate = base_lr * lr_decay ** epoch
    for param_group in optimiter.param_groups:
        param_group['lr'] = learning_rate

# 获取1个batch的准确率
def get_acc(output, label):
    _, pred_label = output.max(1)
    num_correct = (pred_label == label). sum().item()
    return num_correct / output.shape[0]

def train(net, train_data, valid_data):
    if torch.cuda.is_available():
        net = net.cuda()
    prev_time = datetime.now()
    for epoch in range(num_epoches):
        batch_count = 0
        train_loss = 0
        train_acc = 0
        net = net.train()
        # 根据epoch修改学习率
        adjust_lr(optimizer, epoch)
        for im, label in train_data:
            if torch.cuda.is_available():
                im = im.cuda()
                label = label.cuda()
            # forward
            output = net(im)
            loss = criterion(output, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            acc = get_acc(output, label)
            train_acc += acc
            print("--Batch: %d, Loss: %f, Acc: %f" %(batch_count, loss.item(), acc))
            batch_count += 1
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for im, label in valid_data:
                if torch.cuda.is_available():
                    im = im.cuda()
                    label = label.cuda()
                output = net(im)
                loss = criterion(output, label)
                valid_loss += loss.item()
                valid_acc += get_acc(output, label)
            epoch_str = ("Epoch %d.\n Train Loss: %f, Train Acc: %f.\n Valid Loss: %f, Valid Acc: %f. "
                         % (epoch, train_loss/len(train_data), train_acc/len(train_data), valid_loss/len(valid_data), valid_acc/len(valid_data)))
        else:
            epoch_str = ("Epoch %d. Train Loss: %f, Train Acc:%f"
                         %(epoch, train_loss/len(train_data), train_acc/len(train_data)))

        prev_time = cur_time
        print(epoch_str + time_str)
        if epoch + 1 == num_epoches:
            model_path = './model'
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(net.state_dict(), model_path + '/mobilenetv2_face_' + str(epoch + 1) + '.pth')

if __name__=='__main__':
    # 获取训练数据
    #train_data, test_data = get_cifar10_dataset()
    train_data, test_data = get_face_classifier_dataset()
    print("获取人脸2分类数据开始训练")
    train(net, train_data, test_data)