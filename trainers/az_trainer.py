"""AlphaZero trainer CLI shim.

Builds the AZ :class:`~regi_py.rl.trainer_loop.Pipeline` (net registry +
:class:`~regi_py.rl.trainer_loop.Paradigm` of the card-space classes the shared game
runners switch on) and runs the shared orchestration loop; all the mp/process logic
lives in ``regi_py.rl.trainer_loop``. The SEPARATE AZ net registry (``get_net`` /
``net_names``) means ``--net`` here can only build a card-space ``BaseNet``.
"""
from regi_py.rl.az.nets import get_net, net_names
from regi_py.rl.az.explorer import (
    AlphaZeroNode,
    simulate_node,
    NetDirectStrategy,
    AZExplorerStrategy,
)
from regi_py.rl.training import (
    RecordingBruteStrategy,
    RecordingAZTeamStrategy,
    infos_from_game,
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

PIPELINE = Pipeline(
    prog="regi-mcts-trainer",
    label="az",
    net_default="basic",
    net_choices=net_names(),
    get_net=get_net,
    paradigm=AZ_PARADIGM,
)


if __name__ == "__main__":
    run_trainer(PIPELINE)
