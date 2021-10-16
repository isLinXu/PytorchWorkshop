import abc
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import defaultdict


class BaseModel(nn.Module):
    def __init__(self, seed=10101):
        self._predictions = defaultdict(list)
        self._epoch = 0
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        super().__init__()

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, x, **kwargs):
        pass

    @classmethod
    @abc.abstractmethod
    def _get_inputs(cls, iterator):
        pass

    @abc.abstractmethod
    def _compute_metrics(self, target_y, pred_y, predictions_are_classes=True, training=True):
        pass

    @abc.abstractmethod
    def fit(self, optimizer, loss_fn, data_loader, validation_data_loader, num_epochs, logger):
        pass

    @classmethod
    def to_np(cls, x):
        if x is None:
            return np.array([])
        # convert Variable to numpy array
        if isinstance(x, Variable):
            return x.data.cpu().numpy()
        else:
            return x.numpy()

    @classmethod
    def to_var(cls, x, use_gpu=True):
        if torch.cuda.is_available() and use_gpu:
            x = x.cuda(async=True)
        return Variable(x)

    @classmethod
    def to_tensor(cls, x):
        # noinspection PyUnresolvedReferences
        tensor = torch.from_numpy(x).float()
        return tensor

    def _accumulate_results(self, target_y, pred_y, loss=None, **kwargs):
        if loss is not None:
            self._predictions["train_loss"].append(loss)
        for k, v in kwargs.items():
            self._predictions[k].extend(v)
        if target_y is not None:
            self._predictions["target"].extend(target_y)
        if pred_y is not None:
            self._predictions["predicted"].extend(pred_y)

    def show_env_info(self):
        print('__Python VERSION:', sys.version)
        print('__CUDA VERSION')
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__Devices')
        print("OS: ", sys.platform)
        print("PyTorch: ", torch.__version__)
        print("Numpy: ", np.__version__)
        use_cuda = torch.cuda.is_available()
        print("CUDA is available", use_cuda)
        print("----------==================------------")
        print(repr(self))

    def _log_data(self, logger, data_dict):
        for tag, value in data_dict.items():
            logger.scalar_summary(tag, value, self._epoch + 1)

    def _log_grads(self, logger):
        for tag, value in self.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, self.to_np(value), self._epoch + 1)
            if value.grad is not None:
                logger.histo_summary(tag + '/grad', self.to_np(value.grad), self._epoch + 1)

    def _log_and_reset(self, logger, data, log_grads=True):
        self._log_data(logger, data)
        if log_grads:
            self._log_grads(logger)

    def save(self, path):
        torch.save(self, path)

    @classmethod
    def load(cls, path):
        """
        Load model
        :param path: string path to file containing model
        :return: instance of this class
        :rtype: BaseModel
        """
        model = torch.load(path)
        if isinstance(model, dict):
            model = model['state_dict']
        return model
