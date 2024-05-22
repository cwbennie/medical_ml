import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from utils import *


# create test neural network using ResNet18 for transfer learning
class TestNet(nn.Module):
    def __init__(self, output=10, training=False):
        super(TestNet, self).__init__()
        # get pre-trained weights for resnet18 and set parameter gradients
        # to false
        if not training:
            resnet = models.resnet18(weights='DEFAULT')
            for param in resnet.parameters():
                param.requires_grad = False
        else:
            resnet = models.resnet18(weights=None)
        layers = list(resnet.children())[:8]
        self.top_model = nn.Sequential(*layers)
        # add final fully connected layer
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, output)

    def forward(self, x):
        x = F.relu(self.top_model(x))
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# create test neural network using ResNet34 for transfer learning
class TrainResNet(nn.Module):
    def __init__(self, intermediate=False, training=False, output=4):
        super(TrainResNet, self).__init__()
        if not training:
            resnet = models.resnet34(weights='DEFAULT')
            for param in resnet.parameters():
                param.requires_grad = False
        else:
            resnet = models.resnet34(weights=None)
        layers = list(resnet.children())[:8]
        self.top_model = nn.Sequential(*layers)
        self.fc = nn.Linear(512, output)
        self.inter = intermediate

    def forward(self, x):
        x = F.relu(self.top_model(x))
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.size(0), -1)
        if self.inter:
            return x
        x = self.fc(x)
        return x


# create test neural network using mobilenet_v3_large for transfer learning
class TrainMobileNet(nn.Module):
    def __init__(self, intermediate=False, training=False, output=4,
                 model_path=False):
        super(TrainMobileNet, self).__init__()
        if not training:
            mobilenet = models.mobilenet_v3_large(weights='DEFAULT')
            for param in mobilenet.parameters():
                param.requires_grad = False
        else:
            mobilenet = models.mobilenet_v3_large(weights=None)
        self.top_model = nn.Sequential(*mobilenet.features)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifer = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(1280, output)
        )
        self.inter = intermediate

    def forward(self, x):
        x = self.top_model(x)
        x = self.global_pool(x)
        if self.inter:
            return x
        x = torch.flatten(x, 1)
        x = self.classifer(x)
        return x


class RotNet(nn.Module):
    def __init__(self, intermediate=False, training=False, output=4,
                 model_path=False, mod_type='resnet'):
        super(RotNet, self).__init__()
        if model_path:
            if mod_type == 'mobilenet':
                resnet = TrainMobileNet()
            else:
                resnet = TrainResNet()
            load_model(resnet, model_path)
            layers = list(resnet.children())[:1]
            if not training:
                resnet = models.resnet34(weights='DEFAULT')
                for param in resnet.parameters():
                    param.requires_grad = False
        else:
            if not training:
                resnet = models.resnet34(weights='DEFAULT')
                for param in resnet.parameters():
                    param.requires_grad = False
            else:
                resnet = models.resnet34(weights=None)
            layers = list(resnet.children())[:8]
        self.top_model = nn.Sequential(*layers)
        self.fc = nn.Linear(512, output)
        self.inter = intermediate
        self.mod_type = mod_type

    def forward(self, x):
        x = self.top_model(x)
        x = F.relu(x)
        if self.inter:
            return x
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MURANet(nn.Module):
    def __init__(self, output=1):
        super(MURANet, self).__init__()
        densenet = models.densenet121(weights='DEFAULT')
        layers = list(densenet.children())[0]
        self.base_densenet = nn.Sequential(*layers)
        self.fc = nn.Linear(1024, output)

    def forward(self, x):
        img = torch.relu(self.base_densenet(x))
        img = nn.AdaptiveAvgPool2d((1, 1))(img)
        img = img.view(img.size(0), -1)
        img = self.fc(img)
        img = torch.sigmoid(img)
        return img


class MURATriangular(nn.Module):
    def __init__(self, output=1):
        super(MURATriangular, self).__init__()
        densenet = models.densenet121(weights='DEFAULT')
        layers = list(densenet.children())[0]
        self.groups = nn.ModuleList([nn.Sequential(*h) for h in [layers[:7], layers[7:]]])
        self.groups.append(nn.Linear(1024, output))

    def forward(self, x):
        for group in self.groups[:-1]:
            x = group(x)
        img = torch.relu(x)
        img = nn.AdaptiveAvgPool2d((1, 1))(img)
        img = img.view(img.size(0), -1)
        img = self.groups[-1](img)
        img = torch.sigmoid(img)
        return img
