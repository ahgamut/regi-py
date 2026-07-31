"""AlphaDouZero (ADZ) trainer CLI shim.

Builds the ADZ :class:`~regi_py.rl.trainer_loop.Pipeline` and runs the shared
orchestration loop; all the mp/process logic lives in ``regi_py.rl.trainer_loop``.
Uses the SEPARATE ADZ net registry (``get_adz_net`` / ``adz_net_names``), so
``--net`` here can never instantiate a card-space ``BaseNet``.
"""
from regi_py.rl.adz.nets import get_adz_net, adz_net_names
from regi_py.rl.adz_training import (
    adz_run_single_game,
    adz_run_brute_game,
    adz_run_team_game,
    adz_test_model,
    adz_improved_gameplay,
)
from regi_py.rl.trainer_loop import Pipeline, run_trainer

PIPELINE = Pipeline(
    prog="regi-adz-trainer",
    label="adz",
    net_default="adzmulti",
    net_choices=adz_net_names(),
    get_net=get_adz_net,
    run_single=adz_run_single_game,
    run_brute=adz_run_brute_game,
    run_team=adz_run_team_game,
    test_model=adz_test_model,
    improved_gameplay=adz_improved_gameplay,
)


if __name__ == "__main__":
    run_trainer(PIPELINE)
