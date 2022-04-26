import torch.nn as nn
import torch.nn.functional as F
import torch

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.leaky_relu(out, negative_slope=0.1)

#定义make_layer
def make_layer(in_channel, out_channel, block_num, stride=1):
    shortcut = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, stride),
        nn.BatchNorm2d(out_channel))
    layers = list()
    layers.append(ResBlock(in_channel, out_channel, stride, shortcut))

    for i in range(1, block_num):
        layers.append(ResBlock(out_channel, out_channel))
    return nn.Sequential(*layers)

def make_layer_2(in_channel, out_channel, block_num, stride=1):
    layers = list()
    layers.append(ResBlock(in_channel, out_channel, stride))

    for i in range(1, block_num):
        layers.append(ResBlock(out_channel, out_channel))
    return nn.Sequential(*layers)


# 堆叠Resnet，见上表所示结构
class EXAMNET(nn.Module):
    def __init__(self, output_num):
        super(EXAMNET, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, stride=2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.layer1 = make_layer(32, 32, 1, stride=1)
        self.layer2 = make_layer(32, 64, 2, stride=2)
        self.layer3 = make_layer(64, 128, 2, stride=2)
        self.layer4 = make_layer(128, 192, 3, stride=2)
        self.layer5 = make_layer(192, 256, 3, stride=2)
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.layer7 = nn.Linear(256, output_num)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(-1, 256)
        out = self.layer7(x)
        return out


# 堆叠Resnet，见上表所示结构
class EXAMNETTest(nn.Module):
    def __init__(self, output_num):
        super(EXAMNETTest, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, stride=2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.layer1 = make_layer(32, 32, 1, stride=1)
        self.layer2 = make_layer(32, 64, 2, stride=2)
        self.layer3 = make_layer(64, 128, 2, stride=2)
        self.layer4 = make_layer(128, 192, 3, stride=2)
        self.layer5 = make_layer(192, 256, 3, stride=2)
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.layer7 = nn.Linear(256, output_num)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(-1, 256)
        out = self.layer7(x)
        out = self.softmax(out)
        return out



if __name__ == "__main__":
    print("Efficient B0 Summary")
    net = EXAMNET(24)
    example = torch.rand(1, 3, 128, 128)
    net.forward(example)