"""GPU inference server for self-play (trainer_server.py).

One CUDA process batches leaf evals from many explorers. Explorers stay CPU-only:
their net's ``predict`` routes to ``predict_remote``, which tensorifies locally and
exchanges fixed-shape tensors through a preallocated shared-memory arena. Only slot
ids cross the queue -- never tensors -- so /dev/shm stays flat and bounded.
"""
import queue

import torch
import torch.multiprocessing as mp

from regi_py import GameState, DummyLog
from regi_py.strats import RandomStrategy


class InferArena:
    def __init__(self, in_fields, out_fields, req_q, events, version, n_slots):
        self.in_fields = in_fields      # {name: shared cpu tensor (n_slots, *shape)}
        self.out_fields = out_fields    # {name: shared cpu tensor (n_slots, *shape)}
        self.req_q = req_q              # mp.Queue of slot ints only
        self.events = events            # list[mp.Event], one per slot
        self.version = version          # mp.Value('i'), bumped on weight publish
        self.n_slots = n_slots

    def client_for(self, slot):
        return InferClient(self, slot)


class InferClient:
    def __init__(self, arena, slot):
        self.arena = arena
        self.slot = slot

    def exchange(self, in_dict):
        s = self.slot
        a = self.arena
        for f, t in in_dict.items():
            a.in_fields[f][s].copy_(t[0])
        a.events[s].clear()
        a.req_q.put(s)
        a.events[s].wait()
        return {f: a.out_fields[f][s] for f in a.out_fields}


def _sample_node(params):
    net = params.net_cls()
    net.eval()
    game = GameState(DummyLog())
    strat = RandomStrategy()
    for _ in range(2):
        game.add_player(strat)
    game.initialize()
    start_phase = game.export_phaseinfo()
    node = params.pipeline.paradigm.node_cls(
        start_phase, net=net, history=[], prior=1.0, trim=False
    )
    return net, node


def build_arena(params, n_slots):
    net, node = _sample_node(params)
    sample_in = params.net_cls.sample_predict_input(node)
    with torch.inference_mode():
        sample_out = net.predict_batch(sample_in)
    in_fields = {
        f: torch.empty((n_slots,) + tuple(t.shape[1:]), dtype=t.dtype).share_memory_()
        for f, t in sample_in.items()
    }
    out_fields = {
        f: torch.empty((n_slots,) + tuple(t.shape[1:]), dtype=t.dtype).share_memory_()
        for f, t in sample_out.items()
    }
    events = [mp.Event() for _ in range(n_slots)]
    return InferArena(in_fields, out_fields, mp.Queue(), events, mp.Value("i", 0), n_slots)


def stop_server(arena):
    arena.req_q.put(None)


def infer_server(shared_model, arena, device, params):
    torch.set_num_threads(params.num_threads)
    net = params.net_cls()
    net.load_state_dict(shared_model.state_dict())
    net.device = device
    net.to(device)
    net.eval()
    seen = -1
    cap = params.infer_batch
    while True:
        ids = [arena.req_q.get()]
        while len(ids) < cap:
            try:
                ids.append(arena.req_q.get_nowait())
            except queue.Empty:
                break
        if None in ids:                      # stop_server sentinel
            break
        v = arena.version.value
        if v != seen:
            net.load_state_dict(shared_model.state_dict())
            net.to(device)
            seen = v
        idx = torch.as_tensor(ids, dtype=torch.long)
        batch = {f: t[idx] for f, t in arena.in_fields.items()}
        out = net.predict_batch(batch)
        for f, o in out.items():
            arena.out_fields[f][idx] = o
        for i in ids:
            arena.events[i].set()
