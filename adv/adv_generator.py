import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
# from models import *
import numpy as np
import attack_generator as attack
import os

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock,num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  num_blocks[0], stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512, num_classes) #512

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock, [2,2,2,2])

def ResNet34():
    return ResNet(ResidualBlock, [3,4,6,3])


parser = argparse.ArgumentParser(description='PyTorch White-box Adversarial Attack Test')
parser.add_argument('--net', type=str, default="resnet18", help="decide which network to use,choose from resnet18, resnet34")
parser.add_argument('--dataset', type=str, default="cifar10", help="choose from cifar10,svhn")
parser.add_argument('--drop_rate', type=float,default=0.0, help='WRN drop rate')
parser.add_argument('--attack_method', type=str,default="dat", help = "choose form: dat and trades")
parser.add_argument('--model_path', default='./Res18_model/net_150.pth', help='model for white-box attack evaluation')
parser.add_argument('--method',type=str,default='dat',help='select attack setting following DAT or TRADES')

args = parser.parse_args()

transform_test = transforms.Compose([transforms.ToTensor(),])

print('==> Load Test Data')
if args.dataset == "cifar10":
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
if args.dataset == "svhn":
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

print('==> Load Model')
if args.net == "resnet18":
    model = ResNet18().cuda()
    net = "resnet18"
if args.net == "resnet34":
    model = ResNet34().cuda()
    net = "resnet34"


ckpt = torch.load(args.model_path)
model.load_state_dict(ckpt)


model.eval()
print('==> Generate adversarial sample')


PATH_DATA='./adv/Adv_data/cifar10/RN18'


X_adv=attack.adv_generate(model, test_loader, perturb_steps=20, epsilon=8./255, step_size=8./255 / 10, loss_fn="cent", category="Madry", rand_init=True)
os.makedirs(PATH_DATA)
np.save(os.path.join(PATH_DATA, 'Adv_cifar_PGD20_eps8.npy'), X_adv)
