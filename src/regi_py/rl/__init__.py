from .basicnet import BasicNet
from .az.explorer import NetDirectStrategy, AZExplorerStrategy
from .adz.explorer import ADZDirectStrategy, ADZExplorerStrategy

# Exposed through ``regi_py.get_strategy_map()`` (rl_mods=True). These are the
# NN-net-backed strategy classes; they need a trained net to instantiate, so a
# caller builds a configured instance via ``make_net_strategy`` below rather than
# the zero-arg ``strategy_map[name]()`` path used for the torch-free strategies.
STRATEGY_LIST = [
    NetDirectStrategy,
    AZExplorerStrategy,
    ADZDirectStrategy,
    ADZExplorerStrategy,
]


def make_net_strategy(name, iters, weights_path):
    """Build an NN-net-backed strategy from a parsed recommender spec.

    Mirrors the webapp's ``NAME-ITERS`` ("reco_bot") grammar: ``iters == 0`` ->
    a search-free Direct-net strategy, ``iters > 0`` -> a net-guided Explorer.
    AZ vs ADZ is resolved by which net registry holds ``name``. torch and the
    net registries are imported LAZILY so merely importing this package (for the
    ``STRATEGY_LIST`` names) does not force a net build.
    """
    if not weights_path:
        raise ValueError(f"NN net strategy {name!r} requires a weights path")

    import torch  # lazy: only imported when an NN net is actually built
    from regi_py.rl.az.nets import net_names, get_net
    from regi_py.rl.adz.nets import adz_net_names, get_adz_net

    if name in net_names():
        cls, paradigm = get_net(name), "az"
    elif name in adz_net_names():
        cls, paradigm = get_adz_net(name), "adz"
    else:
        choices = ", ".join(net_names() + adz_net_names())
        raise ValueError(f"unknown NN net {name!r}; choices: {choices}")

    net = cls()
    net.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    net.eval()

    # iters decides search-free Direct-net vs net-guided Explorer (the reco_bot
    # rule); the paradigm decides which explorer module supplies them.
    if paradigm == "adz":
        from regi_py.rl.adz.explorer import ADZDirectStrategy, ADZExplorerStrategy

        return (
            ADZDirectStrategy(net)
            if iters == 0
            else ADZExplorerStrategy(net, iterations=iters)
        )
    from regi_py.rl.az.explorer import NetDirectStrategy, AZExplorerStrategy

    return (
        NetDirectStrategy(net)
        if iters == 0
        else AZExplorerStrategy(net, iterations=iters)
    )
