import torch
import random


from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, ConcatDataset


class ShardBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.current_size = 0
        self.shards = []  # list[TensorDataset]

    def add(self, dct):
        ds = TensorDataset(
            dct["location"],
            dct["used_pile"],
            dct["value"],
            dct["keepyness"],
            dct["atk_probs"],
            dct["attacking"],
        )
        if self.current_size + len(ds) < self.capacity:
            self.current_size += len(ds)
            self.shards.append(ds)
            return None
        else:
            i = random.randrange(len(self.shards))
            old = self.shards[i]
            self.shards[i] = ds
            self.current_size += len(ds) - len(old)
            return old

    def dataset(self):
        return ConcatDataset(self.shards)

    def __len__(self):
        return self.current_size


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


class AZDataLoader(DataLoader):
    pass
