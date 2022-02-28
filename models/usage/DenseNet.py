from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

__all__ = ['DenseNet121']

class DenseNet121(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(DenseNet121, self).__init__()
        self.loss = loss
        # densenet121 = torchvision.models.densenet121(pretrained=True)
        densenet121 = torchvision.models.densenet121()
        self.base = densenet121.features
        self.classifier = nn.Linear(1024, num_classes)
        self.feat_dim = 1024 # feature dimension

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        # print('f.shape',f.shape)
        y = self.classifier(f)
        return y



