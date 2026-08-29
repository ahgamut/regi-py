import torch


class ShardBuffer:
    """A bounded, preallocated replay buffer of training-example ROWS.

    Each incoming dict is one game's training tensors (keyed by the net's
    ``TRAIN_FIELDS``, passed in as ``train_fields``); ``add`` COPIES its rows into
    trainer-private, fixed-size ring tensors and keeps NO reference to the incoming
    tensors. This is deliberate: the explorer processes send their tensors over an
    mp queue, and torch shares queued CPU tensors through ``/dev/shm`` (a small,
    RAM-backed tmpfs -- 64 MB in containers). The old list-of-``TensorDataset``
    design retained those shared tensors for their whole buffer residency, so the
    buffer lived in ``/dev/shm`` and overflowed it with "unable to mmap ...: Cannot
    allocate memory". Copying once at ``add`` releases each shared segment the
    instant ``drain`` drops the dict, so the long-lived buffer is ordinary private
    heap and ``/dev/shm`` only ever holds transient in-flight queue traffic. See the
    mp-shared-memory design notes.

    Eviction is FIFO oldest-row (ring overwrite), not the old random-whole-game
    policy. The buffer preallocates the full ``capacity`` (in rows) up front, so RAM
    is eager but bounded and predictable (an over-large ``capacity`` OOMs loudly at
    startup, not subtly in ``/dev/shm``).
    """

    def __init__(self, capacity, train_fields):
        self.capacity = capacity
        self.train_fields = train_fields
        self.fields = None  # lazily allocated on first add (learns shapes/dtypes)
        self.pos = 0  # next write cursor into the ring
        self.current_size = 0  # number of valid rows (<= capacity)

    def _alloc(self, dct):
        # learn each field's per-row shape + dtype from the first shard; a row is
        # dim 0 of every field tensor (tensorify_training stacks one row/decision).
        self.fields = [
            torch.empty((self.capacity, *dct[k].shape[1:]), dtype=dct[k].dtype)
            for k in self.train_fields
        ]

    def add(self, dct):
        if self.fields is None:
            self._alloc(dct)
        nrows = dct[self.train_fields[0]].shape[0]
        if nrows == 0:
            return None
        if nrows >= self.capacity:
            # a single game with more rows than the whole buffer: keep the last
            # ``capacity`` rows, filling the ring exactly once.
            for f, k in enumerate(self.train_fields):
                self.fields[f].copy_(dct[k][nrows - self.capacity :])
            self.pos = 0
            self.current_size = self.capacity
            return None
        start = self.pos
        end = start + nrows
        if end <= self.capacity:
            for f, k in enumerate(self.train_fields):
                self.fields[f][start:end].copy_(dct[k])
        else:  # wrap: split the rows across the end of the ring and the front
            first = self.capacity - start
            for f, k in enumerate(self.train_fields):
                src = dct[k]
                self.fields[f][start:].copy_(src[:first])
                self.fields[f][: nrows - first].copy_(src[first:])
        self.pos = end % self.capacity
        self.current_size = min(self.current_size + nrows, self.capacity)
        return None

    def sample_batch(self, batch_size):
        """Sample a random minibatch (with replacement, uniform over all valid
        rows) as a tuple of stacked field tensors in the net's ``TRAIN_FIELDS``
        order -- the same layout ``run_epoch`` already consumes. Advanced indexing
        returns fresh contiguous tensors, so nothing shared leaks into the batch.
        """
        if self.current_size == 0:
            raise ValueError("cannot sample from an empty ShardBuffer")
        n = min(batch_size, self.current_size)
        idx = torch.randint(0, self.current_size, (n,))
        return tuple(field[idx] for field in self.fields)

    def __len__(self):
        return self.current_size
