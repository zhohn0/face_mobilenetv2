# 使用数据测试已经训练好的模型
import torch
import torch.nn as nn
from mobilenet import MobileNetV2
from load_data import get_cifar10_dataset
from model_train import get_acc

#定义参数
model_path = "./model/mobilenetv2_100.pth"

net = MobileNetV2()
if torch.cuda.is_available():
    net = net.cuda()
net.load_state_dict(torch.load(model_path))
# 与train相同的损失函数
criterion = nn.CrossEntropyLoss()

def t_model(net, test_data):
    net = net.eval()
    test_loss = 0
    test_acc = 0
    for im, label in test_data:
        if torch.cuda.is_available():
            im = im.cuda()
            label = label.cuda()
        output = net(im)
        loss = criterion(output, label)
        test_loss += loss.item()
        acc = get_acc(output, label)
        test_acc += acc
        print("[Loss: %f][Acc: %f]" % (loss, acc))
    print("On Test Data:[Loss: %f][Acc: %f]" % (test_loss/len(test_data), test_acc/len(test_data)))

if __name__ == '__main__':
    # 获取数据
    train_data, test_data = get_cifar10_dataset()
    t_model(net, test_data)

