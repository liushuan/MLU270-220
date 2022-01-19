
import torch.nn as nn
import torch.nn.functional as F
import torch


class ONET(nn.Module):
    def __init__(self):
        super(ONET, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0)
        self.prelu1 = nn.PReLU(16)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.prelu2 = nn.PReLU(32)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0)
        self.prelu3 = nn.PReLU(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=0)
        self.prelu4 = nn.PReLU(64)

        self.conv5 = nn.Linear(576, 128)
        self.prelu5 = nn.PReLU(128)

        self.ip2 = nn.Linear(128 , 10)
        self.ip3 = nn.Linear(128 , 3)


    def forward(self, x):

        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)


        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)

        #print("input:", x)

        x = self.conv4(x)
        x = self.prelu4(x)

        x = x.view(x.size(0), -1)

        x = self.conv5(x)
        x = self.prelu5(x)



        x1 = self.ip2(x)
        x2 = self.ip3(x)
        return x1, x2



