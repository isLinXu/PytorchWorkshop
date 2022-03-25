import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.onnx
from torch import onnx

from timodels import *
from utils.datasets import *
from utils.utils import *
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', type=str, default='tomofun.yaml',
                    help='Specify yaml file (default: tomofun.yaml)')
    parser.add_argument('--model', type=str, default='Convnext_tiny',
                    help='Specify model (default: Convnext_tiny)')
    parser.add_argument('--model_saved_path', type=str, default='workdirs',
                    help='Specify the file path where the model is saved (default: workdirs)')
    parser.add_argument('--model_weights', type=str, default='best.pth',
                    help='Specify the weights where the model is saved (default: best.pth)')

    parser.add_argument('--cuda_num', type=int, default=0,
                        help='number of CUDA (default: 0)')
    parser.add_argument('--local_rank', type=int, default=0,
                    help='number of local_rank for distributed training (default: 0)')
    parser.add_argument('--world-size', type=int, default=1,
                    help='number of nodes for distributed training (default: 1)')

    args = parser.parse_args()
    device, rank, world_size = init_distributed_mode(args.local_rank, args.cuda_num)
    print("device: {}, rank: {}, world_size: {}".format(device, rank, world_size))
    
    data_file, classes_info, data_set = load_data_info(args.yaml_file)
    classes_len, classes_names = classes_info[0], classes_info[1]

    model_stats_path = os.path.join(args.model_saved_path, args.model_weights)

    if args.model == "ResNet18":
        model = ResNet18(1, classes_len)
        
    elif args.model == "ResNet34":
        model = ResNet34(1, classes_len)

    elif args.model == "ResNet50":
        model = ResNet50(1, classes_len)

    elif args.model == "ResNet101":
        model = ResNet101(1, classes_len)

    elif args.model == "ResNet152":
        model = ResNet152(1, classes_len)

    elif args.model == "SENet":
        model = SENet(classes_len)

    elif args.model == "DenseNet":
        model = DenseNet(classes_len)

    elif args.model == "Convnext_tiny":
        model = Convnext_tiny(classes_len)

    elif args.model == "Convnext_small":
        model = Convnext_tiny(classes_len)
        
    elif args.model == "Convnext_base":
        model = Convnext_base(classes_len)
        
    elif args.model == "Convnext_large":
        model = Convnext_tiny(classes_len)

    model.load_state_dict(torch.load(model_stats_path, map_location=torch.device(device)))
    model.eval()
    
    batch_size = 1
    input_shape = (1, 64, 500)
    dummy_input = torch.randn(batch_size, *input_shape, requires_grad=True)

    torch.onnx.export(model, dummy_input, "model.onnx", opset_version=10, do_constant_folding=True,
                      input_names = ['Input'], output_names = ['Output'],
                      dynamic_axes={'Input' : {0 : 'batch_size'}, 'Output' : {0 : 'batch_size'}})
              
    print("Model has been converted to ONNX.")
    
    net = onnx.load("model.onnx")
    onnx.checker.check_model(net)
    print("Model is checked.")