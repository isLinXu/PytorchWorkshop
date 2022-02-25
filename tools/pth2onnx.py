import torch
import torch.onnx
import torchvision
# from tinynet import tinynet
import os


def pth_to_onnx(model, input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0

    model.load_state_dict(torch.load(checkpoint))  # 初始化权重
    model.eval()
    # model.to(device)

    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names,
                      output_names=output_names)  # 指定模型的输入，以及onnx的输出路径
    print("Exporting .pth model to onnx model has been successful!")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    checkpoint = '/home/hxzh02/文档/ModelsLIst/alexnet-owt-4df8aa71.pth'
    onnx_path = '/home/hxzh02/文档/ModelsLIst/alexnet-owt-4df8aa71.onnx'

    # alexnet input(1,3,224,224)
    input = torch.randn(1, 3, 224, 224)
    model = torchvision.models.alexnet()

    # device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
    pth_to_onnx(model, input, checkpoint, onnx_path)
