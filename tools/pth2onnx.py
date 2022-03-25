import torch
import torch.onnx
import torchvision.models as models
from torchsummary import summary
from torch import onnx

import os


def pth_to_onnx(model, input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0
    # 初始化权重
    model.load_state_dict(torch.load(checkpoint), strict=False)

    model.eval()
    # model.to(device)

    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names,
                      output_names=output_names)  # 指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")


def torch_to_onnx(model, path, batch_size, input_shape):
    x = torch.randn(batch_size, *input_shape, requires_grad=True)

    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    print(f"model saved to {path}")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def load_onnx_model(path):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    return model


if __name__ == '__main__':
    print(torch.__version__)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    checkpoint = '/home/hxzh02/文档/ModelsLIst/VggNet/vgg16-397923af.pth'
    onnx_path = '/home/hxzh02/文档/ModelsLIst/VggNet/vgg16-397923af.onnx'
    #
    # checkpoint = '/home/hxzh02/文档/ModelsLIst/VggNet/vgg16-397923af.pth'
    # onnx_path = '/home/hxzh02/文档/ModelsLIst/VggNet/vgg16-397923af.onnx'

    # Print Summary
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = models.shufflenet_v2_x0_5().to(device)
    # summary(model, (3, 224, 224))

    # Model list
    input_size = torch.randn(1, 3, 224, 224)
    # model = models.alexnet()
    model = models.vgg16()
    # model = models.GoogLeNet()
    # model = models.Inception3()
    # model = models.densenet121()
    # model = models.resnet18()
    # model = models.squeezenet1_0()
    # model = models.resnext50_32x4d()
    # model = models.shufflenet_v2_x0_5()
    # model = models.wide_resnet50_2()
    # model = models.regnet_y_400mf()
    # model = models.efficientnet_b0()
    # model = models.mnasnet0_5()
    # model = models.convnext_tiny()
    # model = models.vit_b_16()

    # Trans Model to ONNX
    pth_to_onnx(model, input_size, checkpoint, onnx_path)
