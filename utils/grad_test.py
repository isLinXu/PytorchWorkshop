import torch
from torch.nn import functional as F
from torch import nn, optim


def main():
    x = torch.ones([])
    w = torch.tensor(2., requires_grad=True)
    y = torch.tensor(4.)

    w2 = w.detach().clone()
    w2.requires_grad_()

    w3 = w.clone().detach()
    w3.requires_grad_()

    w4 = w.detach()
    w4.requires_grad_()

    # w will affect w5, w5 will not affect w!
    w5 = w.clone()
    w6 = w

    pred2 = w2 * x
    pred3 = w3 * x
    pred4 = w4 * x
    pred5 = w5 * x
    pred6 = w6 * x

    loss2 = F.mse_loss(pred2, y)
    loss3 = F.mse_loss(pred3, y)
    loss4 = F.mse_loss(pred4, y)
    loss5 = F.mse_loss(pred5, y)
    loss6 = F.mse_loss(pred6, y)

    grad = torch.autograd.grad(loss2, [w, w2, w3, w4, w5, w6], allow_unused=True)
    print(grad)

    grad = torch.autograd.grad(loss3, [w, w2, w3, w4, w5, w6], allow_unused=True)
    print(grad)

    grad = torch.autograd.grad(loss4, [w, w2, w3, w4, w5, w6], allow_unused=True)
    print(grad)

    # clone has gradient connection with w and w5, w6
    grad = torch.autograd.grad(loss5, [w, w2, w3, w4, w5, w6], allow_unused=True)
    print(grad)
    # = symbol has gradient connection surely, with w and w6, No with w5
    grad = torch.autograd.grad(loss6, [w, w2, w3, w4, w5, w6], allow_unused=True)
    print(grad)


if __name__ == '__main__':
    main()
