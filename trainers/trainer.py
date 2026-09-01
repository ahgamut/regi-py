"""Unified self-play trainer CLI for both paradigms (AZ + ADZ).

One script for both stacks; the paradigm is inferred from ``--net``. The AlphaZero
(card-space grid) and AlphaDouZero (candidate-scoring) net registries are DISJOINT, so
the net name alone is unambiguous: ``--net basic`` (or attntrunk/cardtx/percardmlp/mixer/
movetoken) runs AZ; ``--net adzmulti`` (or adzpool) runs ADZ. All mp/process logic lives
in ``regi_py.rl.trainer_loop``; this file only wires each paradigm's registry + the
:class:`~regi_py.rl.trainer_loop.Paradigm` bundle the shared game runners switch on.

  python -m trainers.trainer --net basic    --num-episodes N ...   # AlphaZero
  python -m trainers.trainer --net adzmulti  --num-episodes N ...   # AlphaDouZero
"""
from regi_py.rl.az.nets import get_net, net_names
from regi_py.rl.az.explorer import (
    AlphaZeroNode,
    simulate_node,
    NetDirectStrategy,
    AZExplorerStrategy,
)
from regi_py.rl.adz.nets import get_adz_net, adz_net_names
from regi_py.rl.adz.explorer import (
    ADZNode,
    adz_simulate_node,
    ADZDirectStrategy,
    ADZExplorerStrategy,
)
from regi_py.rl.training import (
    RecordingBruteStrategy,
    RecordingAZTeamStrategy,
    infos_from_game,
)
from regi_py.rl.adz_training import (
    RecordingADZBruteStrategy,
    RecordingADZTeamStrategy,
    adz_infos_from_game,
)
from regi_py.rl.trainer_loop import Pipeline, Paradigm, run_trainer

AZ_PARADIGM = Paradigm(
    node_cls=AlphaZeroNode,
    simulate_fn=simulate_node,
    brute_recorder=RecordingBruteStrategy,
    team_recorder=RecordingAZTeamStrategy,
    infos_fn=infos_from_game,
    direct_strat=NetDirectStrategy,
    explorer_strat=AZExplorerStrategy,
)

ADZ_PARADIGM = Paradigm(
    node_cls=ADZNode,
    simulate_fn=adz_simulate_node,
    brute_recorder=RecordingADZBruteStrategy,
    team_recorder=RecordingADZTeamStrategy,
    infos_fn=adz_infos_from_game,
    direct_strat=ADZDirectStrategy,
    explorer_strat=ADZExplorerStrategy,
)

AZ = Pipeline(
    prog="regi-trainer",
    label="az",
    net_default="basic",  # default net -> default paradigm (AZ)
    net_choices=net_names(),
    get_net=get_net,
    paradigm=AZ_PARADIGM,
)

ADZ = Pipeline(
    prog="regi-trainer",
    label="adz",
    net_default="adzmulti",
    net_choices=adz_net_names(),
    get_net=get_adz_net,
    paradigm=ADZ_PARADIGM,
)


if __name__ == "__main__":
    run_trainer([AZ, ADZ])
