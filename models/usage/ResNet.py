# -*- coding: utf-8 -*-
from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
import torchvision

__all__ = ['ResNet50', 'ResNet50M','ResNet152','ResNet18','ResNet50_Gender_BOT','ResNet50_BOT_MultiTask','ResNet101_BOT_MultiTask']

class ResNet50(nn.Module):
	def __init__(self, num_classes, loss={'xent'}, **kwargs):
		super(ResNet50, self).__init__()
		self.loss = loss
		resnet50 = torchvision.models.resnet50(pretrained=True)
		self.base = nn.Sequential(*list(resnet50.children())[:-2]) # 去掉最后两层，fc和pooling
		self.classifier = nn.Linear(2048, num_classes)
		self.feat_dim = 2048 # feature dimension

	def forward(self, x):
		x = self.base(x)
		x = F.avg_pool2d(x, x.size()[2:])
		f = x.view(x.size(0), -1)
		# if not self.training:
		#     return f
		y = self.classifier(f)

		if self.loss == {'xent'}:
			return y
		elif self.loss == {'xent', 'htri'}:
			return y, f
		elif self.loss == {'cent'}:
			return y, f
		elif self.loss == {'ring'}:
			return y, f
		else:
			raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet50M(nn.Module):
	"""ResNet50 + mid-level features.

	Reference:
	Yu et al. The Devil is in the Middle: Exploiting Mid-level Representations for
	Cross-Domain Instance Matching. arXiv:1711.08106.
	"""
	def __init__(self, num_classes=0, loss={'xent'}, **kwargs):
		super(ResNet50M, self).__init__()
		self.loss = loss
		resnet50 = torchvision.models.resnet50(pretrained=True)
		# resnet50 = torchvision.models.resnet50()
		base = nn.Sequential(*list(resnet50.children())[:-2])
		self.layers1 = nn.Sequential(base[0], base[1], base[2])
		self.layers2 = nn.Sequential(base[3], base[4])
		self.layers3 = base[5]
		self.layers4 = base[6]
		self.layers5a = base[7][0]
		self.layers5b = base[7][1]
		self.layers5c = base[7][2]
		self.fc_fuse = nn.Sequential(nn.Linear(4096, 1024), nn.BatchNorm1d(1024), nn.ReLU())
		self.classifier = nn.Linear(3072, num_classes)
		self.feat_dim = 3072 # feature dimension

	def forward(self, x):
		x1 = self.layers1(x)
		x2 = self.layers2(x1)
		x3 = self.layers3(x2)
		x4 = self.layers4(x3)
		x5a = self.layers5a(x4)
		x5b = self.layers5b(x5a)
		x5c = self.layers5c(x5b)

		x5a_feat = F.avg_pool2d(x5a, x5a.size()[2:]).view(x5a.size(0), x5a.size(1))  # shape:[1,2048]
		# print('x5a_feat',x5a_feat.shape)
		x5b_feat = F.avg_pool2d(x5b, x5b.size()[2:]).view(x5b.size(0), x5b.size(1))  # shape:[1,2048]
		# print('x5b_feat',x5b_feat.shape)
		x5c_feat = F.avg_pool2d(x5c, x5c.size()[2:]).view(x5c.size(0), x5c.size(1))  # shape:[1,2048]
		# print('x5c_feat',x5c_feat.shape)

		midfeat = torch.cat((x5a_feat, x5b_feat), dim=1)  # shape:[1,4096]
		# print('midfeat',midfeat.shape)
		midfeat = self.fc_fuse(midfeat)  # shape:[1,1024]
		# print('midfeat fc_fuse',midfeat.shape)
		combofeat = torch.cat((x5c_feat, midfeat), dim=1)  # shape：[1,3072]
		# print('combofeat',combofeat.shape)
		prelogits = self.classifier(combofeat)  # shape:[1,num_class]
		if not self.training:
			# print('test')
			return prelogits
		# print('prelogits.shape',prelogits.shape)
		if self.loss == {'xent'}:
			# print('train xent')
			return prelogits

		else:
			raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet152(nn.Module):
	def __init__(self, num_classes, loss={'xent'}, **kwargs):
		super(ResNet152, self).__init__()
		self.loss = loss
		resnet152 = torchvision.models.resnet152(pretrained=True)
		self.base = nn.Sequential(*list(resnet152.children())[:-2])
		self.classifier = nn.Linear(2048, num_classes)
		self.feat_dim = 2048  # feature dimension

	def forward(self, x):
		x = self.base(x)
		x = F.avg_pool2d(x, x.size()[2:])
		f = x.view(x.size(0), -1)
		y = self.classifier(f)
		return y

class ResNet18(nn.Module):
	def __init__(self, num_classes, loss={'xent'}, **kwargs):
		super(ResNet18, self).__init__()
		self.loss = loss
		resnet18 = torchvision.models.resnet18(pretrained=True)
		self.base = nn.Sequential(*list(resnet18.children())[:-2])
		self.classifier = nn.Linear(512, num_classes)
		self.feat_dim = 512  # feature dimension

	def forward(self, x):
		x = self.base(x)
		x = F.avg_pool2d(x, x.size()[2:])
		f = x.view(x.size(0), -1)

		y = self.classifier(f)
		return y

class ResNet50_Gender_BOT(nn.Module):
	def __init__(self, classes = 2, loss={'xent'}, **kwargs):
		super(ResNet50_Gender_BOT, self).__init__()
		self.loss = loss
		resnet50 = torchvision.models.resnet50(pretrained=True)
		self.base = nn.Sequential(*list(resnet50.children())[:-2])  # 去掉最后两层，fc和pooling
		self.gender_classifier = nn.Linear(2048, classes)
		self.feat_dim = 2048  # feature dimension

	def forward(self, x):
		x = self.base(x)
		x = F.avg_pool2d(x, x.size()[2:])
		f = x.view(x.size(0), -1)
		gender = self.gender_classifier(f)

		return gender

class ResNet50_BOT_MultiTask(nn.Module):
	def __init__(self, classes = 2 , loss={'xent'}, **kwargs):
		super(ResNet50_BOT_MultiTask, self).__init__()
		self.loss = loss
		resnet50 = torchvision.models.resnet50(pretrained=True)
		self.base = nn.Sequential(*list(resnet50.children())[:-2])  # 去掉最后两层，fc和pooling

		self.gender_classifier = nn.Linear(2048, classes)
		self.staff_classifier = nn.Linear(2048, classes)
		self.customer_classifier = nn.Linear(2048, classes)
		self.stand_classifier = nn.Linear(2048, classes)
		self.sit_classifier = nn.Linear(2048, classes)
		self.phone_classifier = nn.Linear(2048, classes)

		self.feat_dim = 2048  # feature dimension

	def forward(self, x):
		x = self.base(x)
		x = F.avg_pool2d(x, x.size()[2:])
		f = x.view(x.size(0), -1)
		# print('f.shape',f.shape)
		gender = self.gender_classifier(f)
		staff = self.staff_classifier(f)
		customer = self.customer_classifier(f)
		stand = self.stand_classifier(f)
		sit = self.sit_classifier(f)
		phone = self.phone_classifier(f)
		return gender, staff, customer, stand, sit, phone

class ResNet101_BOT_MultiTask(nn.Module):
	def __init__(self, classes=2, loss={'xent'}, **kwargs):
		super(ResNet101_BOT_MultiTask, self).__init__()
		self.loss = loss
		resnet101 = torchvision.models.resnet101(pretrained=True)
		self.base = nn.Sequential(*list(resnet101.children())[:-2])

		self.gender_classifier = nn.Linear(2048, classes)
		self.staff_classifier = nn.Linear(2048, classes)
		self.customer_classifier = nn.Linear(2048, classes)
		self.stand_classifier = nn.Linear(2048, classes)
		self.sit_classifier = nn.Linear(2048, classes)
		self.phone_classifier = nn.Linear(2048, classes)

		self.feat_dim = 2048  # feature dimension

	def forward(self, x):
		x = self.base(x)
		x = F.avg_pool2d(x, x.size()[2:])
		f = x.view(x.size(0), -1)

		gender = self.gender_classifier(f)
		staff = self.staff_classifier(f)
		customer = self.customer_classifier(f)
		stand = self.stand_classifier(f)
		sit = self.sit_classifier(f)
		phone = self.phone_classifier(f)
		return gender, staff, customer, stand, sit, phone



