import torch
import torch.nn as nn
import torch.hub
import timm

def SENet(num_classes):
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load('moskomule/senet.pytorch','se_resnet20')
    model.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.fc = nn.Sequential(nn.Linear(64, 64),
                             nn.LeakyReLU(),
                             nn.Dropout(0.5),
                             nn.Linear(64, num_classes))
    return model

def DenseNet(num_classes):
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.classifier = nn.Sequential(nn.Linear(1024, 128),
                                     nn.LeakyReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(128, num_classes))
    return model
    