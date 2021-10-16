import os
import argparse
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--rank", type=int, default=0)
args = parser.parse_known_args()[0]


def opt(): pass

if torch.cuda.is_available():
    opt.dtype = torch.half
    opt.device = torch.device('cuda:{}'.format(args.local_rank))
    torch.cuda.set_device(args.local_rank)
    opt.cuda = True
    opt.fp16 = 2
    from apex import amp
else:
    opt.device = torch.device('cpu')
    opt.dtype = torch.float
    opt.cuda = False
    opt.fp16 = 0
    num_threads = torch.multiprocessing.cpu_count() - 1
    if num_threads > 1:
        torch.set_num_threads(num_threads)
    amp = None
print('Using device ' + str(opt.device))
print('Using default dtype ' + str(opt.dtype))

upTruncBy8 = lambda x: (-x & -8 ^ -1) + 1


def getWriter(name='.', writer=None):
    import torchvision.utils as vutils
    if not writer:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(name)

    def write(scalars={}, images={}, histograms={}, n=0):
        for key in scalars:
            writer.add_scalar(key, scalars[key], n)
        for key in images:
            x = vutils.make_grid(images[key])
            writer.add_image(key, x, n)
        for key in histograms:
            writer.add_histogram(key, histograms[key].data, n)

    return write, writer
