from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
import torch
import random


class PUCTDataset(IterableDataset):
    def __init__(self, maxsize=128):
        super().__init__()
        self.samples = []
        self.maxsize = maxsize

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        for x in self.samples:
            yield x

    def add_game(self, net, infos):
        pieces = net.tensorify(infos)
        if len(pieces) + len(self.samples) > self.maxsize:
            N = self.maxsize - len(pieces)
            random.shuffle(self.samples)
            self.samples = self.samples[:N] + pieces
        else:
            self.samples = self.samples + pieces


def collate_dict(objs):
    res = dict()
    for k in objs[0].keys():
        res[k] = torch.cat([obj[k] for obj in objs])
    return res


class PUCTDataLoader(DataLoader):
    def __init__(self, **kwargs):
        super().__init__(collate_fn=collate_dict, **kwargs)
