import random

import torch
from torch.utils.data import TensorDataset

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

    def sample_batch(self, batch_size):
        """Sample a random minibatch (with replacement, uniform over all stored
        rows) as a tuple of stacked field tensors in ``BasicNet.TRAIN_FIELDS``
        order -- the same layout a ``DataLoader`` over the shards would yield, so
        ``run_epoch`` consumes it unchanged. All shards live in RAM, so this
        avoids the per-epoch DataLoader/worker rebuild entirely.
        """
        if not self.shards:
            raise ValueError("cannot sample from an empty ShardBuffer")
        sizes = [len(s) for s in self.shards]
        n = min(batch_size, sum(sizes))
        # picking a shard weighted by its size, then a uniform row within it, is
        # exactly uniform sampling over the buffer's rows
        num_fields = len(self.shards[0].tensors)
        cols = [[] for _ in range(num_fields)]
        for si in random.choices(range(len(self.shards)), weights=sizes, k=n):
            row = random.randrange(sizes[si])
            for f, t in enumerate(self.shards[si].tensors):
                cols[f].append(t[row])
        return tuple(torch.stack(col, dim=0) for col in cols)

    def __len__(self):
        return self.current_size
