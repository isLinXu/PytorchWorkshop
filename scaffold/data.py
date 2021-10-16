from .common import *
from torch.utils.data import Dataset, DataLoader

dataLength = {'train': 4096, 'val': 256, 'test': 256}
vocabsize = 8

class Data(Dataset):
    def __init__(self, path, opt, **kwargs):
        super(Data, self).__init__()
        l = dataLength[path]
        self.lens = torch.ones((l,), dtype=torch.long) * 4
        self.mask = torch.ones((l, 4), dtype=torch.uint8)
        self.data = torch.randint(vocabsize - 1, (l, 4), dtype=torch.long) + 1
        self.count = l
    def __len__(self):
        return self.count
    # input, label, length, mask
    def __getitem__(self, ind):
        x = self.data[ind]
        return x, x, self.lens[ind], self.mask[ind]

newLoader = lambda path, opt, *args, **kwargs: DataLoader(Data(path, opt), *args, **kwargs)
