import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __getitem__(self, item):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class ToTensor:
    def __call__(self, x):
        result = {k: torch.from_numpy(v).float() for k, v in x.items()}
        return result
