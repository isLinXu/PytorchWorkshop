from __future__ import absolute_import
from torch import nn

from .ResNet import *
from .DenseNet import *
from .MuDeep import *
from .HACNN import *
from .SqueezeNet import *
from .MobileNet import *
from .ShuffleNet import *
from .Xception import *
from .Inception import *
from .SEResNet import *


__factory = {
    'resnet50': ResNet50,
	'resnet18': ResNet18,
	'ResNet50_Gender_BOT': ResNet50_Gender_BOT,
	'ResNet50_BOT_MultiTask': ResNet50_BOT_MultiTask,
    'densenet121': DenseNet121,
    'resnet50m': ResNet50M,
    'resnet152': ResNet152,
    'mudeep': MuDeep,
    'hacnn': HACNN,
    'squeezenet': SqueezeNet,
    'mobilenet': MobileNetV2,
    'shufflenet': ShuffleNet,
    'xception': Xception,
	'Xception_BOT_MultiTask': Xception_BOT_MultiTask,
    'inceptionv4': InceptionV4ReID,
    'seresnet50': SEResNet50,
	'SEResNet50_BOT_multi':SEResNet50_BOT_multi,
	'MobileNetV2_bot':MobileNetV2_bot,
	'ResNet101_BOT_MultiTask':ResNet101_BOT_MultiTask
}


def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)