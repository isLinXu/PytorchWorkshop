import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


# # Define a convolution neural network
# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(12)
#         self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(12)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
#         self.bn4 = nn.BatchNorm2d(24)
#         self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
#         self.bn5 = nn.BatchNorm2d(24)
#         self.fc1 = nn.Linear(24 * 10 * 10, 10)
#
#     def forward(self, input):
#         output = F.relu(self.bn1(self.conv1(input)))
#         output = F.relu(self.bn2(self.conv2(output)))
#         output = self.pool(output)
#         output = F.relu(self.bn4(self.conv4(output)))
#         output = F.relu(self.bn5(self.conv5(output)))
#         output = output.view(-1, 24 * 10 * 10)
#         output = self.fc1(output)
#
#         return output


# Instantiate a neural network model
# model = torchvision.models.alexnet()
# model = Network()

import torch.onnx

#Function to Convert to ONNX
def Convert_ONNX(input_size):

    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, input_size, requires_grad=True)

    # Export the model
    torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         "ImageClassifier.onnx",       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         opset_version=10,    # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                                'modelOutput' : {0 : 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')


if __name__ == "__main__":
    # Let's build our model
    # train(5)
    # print('Finished Training')

    # Test which classes performed well
    # testAccuracy()

    # Let's load the model we just created and test the accuracy per label
    path = "/home/hxzh02/文档/ModelsLIst/alexnet-owt-4df8aa71.pth"
    input_size = (int(224),int(224))
    # model = Network()
    model = torchvision.models.alexnet()

    model.load_state_dict(torch.load(path))

    # Test with batch of images
    # testBatch()
    # Test how the classes performed
    # testClassess()

    # Conversion to ONNX
    Convert_ONNX(input_size)
