import random

from torch.utils.data import TensorDataset, ConcatDataset

from regi_py.rl.basicnet import BasicNet


class ShardBuffer:
    """A bounded replay buffer of per-game ``TensorDataset`` shards.

    Each added dict is one self-play game's training tensors (keyed by
    ``BasicNet.TRAIN_FIELDS``); once full, a random shard is evicted.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.current_size = 0
        self.shards = []  # list[TensorDataset]

    def add(self, dct):
        ds = TensorDataset(*(dct[k] for k in BasicNet.TRAIN_FIELDS))
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
