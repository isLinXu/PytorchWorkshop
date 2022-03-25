import torch
import torch.nn as nn
import timm

def Convnext_tiny(num_classes):
    model =  timm.create_model('convnext_tiny', pretrained=False, num_classes=num_classes, drop_path_rate=0)
    model.stem[0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
    
    return model

def Convnext_small(num_classes):
    model =  timm.create_model('convnext_small', pretrained=False, num_classes=num_classes, drop_path_rate=0)
    model.stem[0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
    
    return model

def Convnext_base(num_classes):
    model =  timm.create_model('convnext_base', pretrained=False, num_classes=num_classes, drop_path_rate=0)
    model.stem[0] = nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
    
    return model
    
def Convnext_large(num_classes):
    model =  timm.create_model('convnext_large', pretrained=False, num_classes=num_classes, drop_path_rate=0)
    model.stem[0] = nn.Conv2d(1, 192, kernel_size=(4, 4), stride=(4, 4))
    
    return model
