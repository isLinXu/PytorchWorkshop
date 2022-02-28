import torch
import torch.onnx
import torchvision.models as models
# from tinynet import tinynet
import os


def pth_to_onnx(model, input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0
    # 初始化权重
    model.load_state_dict(torch.load(checkpoint),strict=False)
    model.eval()
    # model.to(device)

    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names,
                      output_names=output_names)  # 指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    checkpoint = '/home/hxzh02/文档/ModelsLIst/DenseNet/densenet121-a639ec97.pth'
    onnx_path = '/home/hxzh02/文档/ModelsLIst/DenseNet/densenet121-a639ec97.onnx'

    # alexnet input(1,3,224,224)
    input = torch.randn(1, 3, 224, 224)
    # model = models.alexnet()
    # model = models.vgg16()
    # model = models.GoogLeNet()
    # model = models.Inception3()
    # model = models.densenet121()
    model = models.resnet18()
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)


    # device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
    pth_to_onnx(model, input, checkpoint, onnx_path)