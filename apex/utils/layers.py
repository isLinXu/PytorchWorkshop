import torch.nn.functional as F
import torch
import torch.nn as nn
from utils.utils import *


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True,bias=False):
        super(BasicConv,self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):
    # BasicRFB是可以设置stride=2的，可以对特征图进行下采样，但是如果放在yolov3的检测头中，应设置stride为固定的1
    def __init__(self, in_planes, out_planes, stride=1, visual=1, scale=0.1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channel = out_planes
        inter_planes = in_planes//8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),  # 这里使用stride=stride主要
            # 是用来改变RFB模块中RFs的大小，因为RFB模型在SSD框架下有stride=2的RFB模块，stride=2时，inception结构部分
            # 的特征图将变小一倍，对应了RFB论文中Fig.5的RFB_stride2
            BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, 
                      padding=visual, dilation=visual, relu=False)
            )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),  # 这里的
            # stride=stride同self.branch0中的stride一样，branch1中的前两个BasicConv属于Inception结构部分，
            # 其中第一个BasiceConv用来降低通道数，第二个BasicConv就来输出不同尺寸的RFs
            BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*visual,
                      dilation=3*visual,relu=False)
            )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1,stride=1),
            BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=5*visual,
                      dilation=5*visual, relu=False)
            )
        
        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1,relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        
        out = torch.cat((x0, x1, x2),1)
        out = self.ConvLinear(out)
        shortcut = self.shortcut(x)
        out = out*self.scale + shortcut
        out = self.relu(out)
        
        return out


class BasicRFB_6_branch_Add_MaxPool(nn.Module):
    # BasicRFB是可以设置stride=2的，可以对特征图进行下采样，但是如果放在yolov3的检测头中，应设置stride为固定的1
    def __init__(self, in_planes, out_planes, press_scale = 1, stride=1, visual=1):
        super(BasicRFB_6_branch_Add_MaxPool, self).__init__()

        inter_planes = in_planes // press_scale
        self.out_channel = out_planes
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=stride),  # 这里使用stride=stride主要
            # 是用来改变RFB模块中RFs的大小，因为RFB模型在SSD框架下有stride=2的RFB模块，stride=2时，inception结构部分
            # 的特征图将变小一倍，对应了RFB论文中Fig.5的RFB_stride2
            BasicConv(inter_planes, in_planes, kernel_size=3, stride=1,
                      padding=visual, dilation=visual, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1)),  # 这里的
            # stride=stride同self.branch0中的stride一样，branch1中的前两个BasicConv属于Inception结构部分，
            # 其中第一个BasiceConv用来降低通道数，第二个BasicConv就来输出不同尺寸的RFs
            BasicConv(inter_planes, in_planes, kernel_size=3, stride=1, padding= visual+1,
                      dilation= visual + 1, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(inter_planes, in_planes, kernel_size=3, stride=1, padding= 2 * visual + 1,
                      dilation= 2 * visual + 1, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=stride, padding=1),
            BasicConv(inter_planes, in_planes, kernel_size=3, stride=1, padding= 3 * visual + 1,
                      dilation= 3 * visual + 1, relu=False)
        )
        self.max_pool_branch = nn.MaxPool2d(23, 1, 11)
        self.ConvLinear = BasicConv(4 * in_planes, 4*in_planes, kernel_size=1, stride=1, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.max_pool_branch(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        out = self.relu(out)
        out_add_max = torch.cat((out, x4, x),1)

        return out_add_max

        
class BasicRFB_a(nn.Module):
    # BasicRFB_a可以用在检测小物体的yolo3层前，当时默认在yolo里面应该不需要设置stride=2（即不需要进行下采样）
    def __init__(self, in_planes, out_planes, stride=1, visual=2, scale=0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        inter_channels = in_planes // 4
        self.out_planes = out_planes

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_channels, kernel_size=1,stride=1),
            BasicConv(inter_channels, inter_channels, kernel_size=3, stride=1,
                      padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_channels, kernel_size=1, stride=1),
            BasicConv(inter_channels, inter_channels, kernel_size=(3,1), stride=stride, padding=(1,0)),
            BasicConv(inter_channels, inter_channels, kernel_size=3, stride=1, padding=3,
                      dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_channels, kernel_size=1, stride=1),
            BasicConv(inter_channels, inter_channels, kernel_size=(1,3), stride=stride, padding=(0,1)),
            BasicConv(inter_channels, inter_channels, kernel_size=3, stride=1, padding=visual+1,
                      dilation=visual+1, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_channels//2, kernel_size=1, stride=1),
            BasicConv(inter_channels//2, (inter_channels//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv((inter_channels//4)*3, inter_channels, kernel_size=(3,1), stride=stride, padding=(1,0)),
            BasicConv(inter_channels, inter_channels, kernel_size=3, stride=1, padding=2*visual+1,
                      dilation=2*visual+1,relu=False)
        )
        self.ConvLinear = BasicConv(4*inter_channels, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        shortcut = self.shortcut(x)
        out = self.scale * out + shortcut
        out = self.relu(out)

        return out

"""
class BasicRFBSA_a(nn.Module):
    # BasicRFB_a可以用在检测小物体的yolo3层前，当时默认在yolo里面应该不需要设置stride=2（即不需要进行下采样）
    def __init__(self, in_planes, out_planes, stride=1, visual=2, scale=0.1):
        super(BasicRFBSA_a, self).__init__()
        self.scale = scale
        inter_channels = in_planes // 4
        self.out_planes = out_planes

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_channels, kernel_size=1,stride=1),
            BasicConv(inter_channels, inter_channels, kernel_size=3, stride=1,
                      padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_channels, kernel_size=1, stride=1),
            BasicConv(inter_channels, inter_channels, kernel_size=(3,1), stride=stride, padding=(1,0)),
            BasicConv(inter_channels, inter_channels, kernel_size=3, stride=1, padding=3,
                      dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_channels, kernel_size=1, stride=1),
            BasicConv(inter_channels, inter_channels, kernel_size=(1,3), stride=stride, padding=(0,1)),
            BasicConv(inter_channels, inter_channels, kernel_size=3, stride=1, padding=visual+1,
                      dilation=visual+1, relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_planes, inter_channels//2, kernel_size=1, stride=1),
            BasicConv(inter_channels//2, (inter_channels//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv((inter_channels//4)*3, inter_channels, kernel_size=(3,1), stride=stride, padding=(1,0)),
            BasicConv(inter_channels, inter_channels, kernel_size=3, stride=1, padding=2*visual+1,
                      dilation=2*visual+1,relu=False)
        )
        self.ConvLinear = BasicConv(4*inter_channels, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        shortcut = self.shortcut(x)
        out = self.scale * out + shortcut
        out = self.relu(out)

        return out
"""

class BasicRFB_a_2b_1_3(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, visual=1, scale=0.1):
        super(BasicRFB_a_2b_1_3, self).__init__()
        inter_planes = in_planes // 4
        self.out_planes = out_planes
        self.scale = scale
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1,
                      padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=1,
                      padding=(0, 1)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=visual+1,
                      dilation=visual+1, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1,
                      padding=(1, 0)),
            BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=visual+1,
                      dilation=visual+1, relu=False)
        )

        self.ConvLinear = BasicConv(3*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class BasicRFB_2b_1_3(nn.Module):
    # BasicRFB是可以设置stride=2的，可以对特征图进行下采样，但是如果放在yolov3的检测头中，应设置stride为固定的1
    def __init__(self, in_planes, out_planes, stride=1, visual=1, scale=0.1):
        super(BasicRFB_2b_1_3, self).__init__()
        self.scale = scale
        self.out_channel = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),  # 这里使用stride=stride主要
            # 是用来改变RFB模块中RFs的大小，因为RFB模型在SSD框架下有stride=2的RFB模块，stride=2时，inception结构部分
            # 的特征图将变小一倍，对应了RFB论文中Fig.5的RFB_stride2
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1,
                      padding=visual, dilation=visual, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),  # 这里的
            # stride=stride同self.branch0中的stride一样，branch1中的前两个BasicConv属于Inception结构部分，
            # 其中第一个BasiceConv用来降低通道数，第二个BasicConv就来输出不同尺寸的RFs
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1,
                      dilation=visual + 1, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, 2*inter_planes, kernel_size = (1, 3), stride=stride, padding=(0, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=visual + 1,
                      dilation=visual + 1, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        shortcut = self.shortcut(x)
        out = out * self.scale + shortcut
        out = self.relu(out)

        return out


class ECA_S_layer(nn.Module):
    def __init__(self, eca_kernel_size):
        super(ECA_S_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=eca_kernel_size, padding=(eca_kernel_size - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_avg = self.avg_pool(x)
        y_max = self.max_pool(x)

        y_avg = self.conv(y_avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y_max = self.conv(y_max.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = y_avg + y_max
        y = self.sigmoid(y)

        return x * y


class ECA_layer(nn.Module):
    def __init__(self, eca_kernel_size):
        super(ECA_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=eca_kernel_size, padding=(eca_kernel_size - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        y = self.avg_pool(x)

        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)

        return x * y



class ECBAM(nn.Module):
    def __init__(self, eca_kernel_size, sa_kernel_size):
        super(ECBAM, self).__init__()
        self.eca_layer = ECA_layer(eca_kernel_size)
        self.sa_layer = SpatialAttention(sa_kernel_size)

    def forward(self, x):

        y = self.eca_layer(x)
        out = self.sa_layer(y)

        return out

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio, sa_kernel_size):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(sa_kernel_size)

    def forward(self, x):
        y = self.ca(x)
        out = self.sa(y)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes//ratio, in_planes, 1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self,sa_kernel_size):
        super(SpatialAttention, self).__init__()
        assert sa_kernel_size in (3, 5, 7)  # TODO 不能设置5，必须是3或者7
        # padding = 3 if k_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, sa_kernel_size, padding=(sa_kernel_size - 1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], 1)
        out = self.conv1(out)

        return x * self.sigmoid(out)


def make_divisible(v, divisor):
    # Function ensures all layers have a channel number that is divisible by 8
    # https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    return math.ceil(v / divisor) * divisor


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    def forward(self, x):
        return x.view(x.size(0), -1)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class FeatureConcat(nn.Module):
    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]


class WeightedFeatureFusion(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers
        self.relu = nn.ReLU(inplace=True)
        if weight:
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    def forward(self, x, outputs):
        # Weights
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            na = a.shape[1]  # feature channels

            # Adjust channels
            if nx == na:  # same shape
                x = x + a
            elif nx > na:  # slice input
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]
        x = self.relu(x)
        return x


class MixConv2d(nn.Module):  # MixConv: Mixed Depthwise Convolutional Kernels https://arxiv.org/abs/1907.09595
    def __init__(self, in_ch, out_ch, k=(3, 5, 7), stride=1, dilation=1, bias=True, method='equal_params'):
        super(MixConv2d, self).__init__()

        groups = len(k)
        if method == 'equal_ch':  # equal channels per group
            i = torch.linspace(0, groups - 1E-6, out_ch).floor()  # out_ch indices
            ch = [(i == g).sum() for g in range(groups)]
        else:  # 'equal_params': equal parameter count per group
            b = [out_ch] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(in_channels=in_ch,
                                          out_channels=ch[g],
                                          kernel_size=k[g],
                                          stride=stride,
                                          padding=k[g] // 2,  # 'same' pad
                                          dilation=dilation,
                                          bias=bias) for g in range(groups)])

    def forward(self, x):
        return torch.cat([m(x) for m in self.m], 1)


# Activation functions below -------------------------------------------------------------------------------------------
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x.mul_(torch.sigmoid(x))


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x.mul_(F.softplus(x).tanh())
