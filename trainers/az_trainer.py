"""AlphaZero trainer CLI shim.

Builds the AZ :class:`~regi_py.rl.trainer_loop.Pipeline` and runs the shared
orchestration loop; all the mp/process logic lives in ``regi_py.rl.trainer_loop``.
The SEPARATE AZ net registry (``get_net`` / ``net_names``) means ``--net`` here can
only build a card-space ``BaseNet``.
"""
from regi_py.rl.az.nets import get_net, net_names
from regi_py.rl.training import (
    run_single_game,
    run_brute_game,
    run_team_game,
    test_model,
    improved_gameplay,
)
from regi_py.rl.trainer_loop import Pipeline, run_trainer

PIPELINE = Pipeline(
    prog="regi-mcts-trainer",
    label="az",
    net_default="basic",
    net_choices=net_names(),
    get_net=get_net,
    run_single=run_single_game,
    run_brute=run_brute_game,
    run_team=run_team_game,
    test_model=test_model,
    improved_gameplay=improved_gameplay,
)


if __name__ == "__main__":
    run_trainer(PIPELINE)
