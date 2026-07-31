"""AlphaDouZero (ADZ) trainer CLI shim.

Builds the ADZ :class:`~regi_py.rl.trainer_loop.Pipeline` (net registry +
:class:`~regi_py.rl.trainer_loop.Paradigm` of the candidate-scoring classes the shared
game runners switch on) and runs the shared orchestration loop; all the mp/process
logic lives in ``regi_py.rl.trainer_loop``. Uses the SEPARATE ADZ net registry
(``get_adz_net`` / ``adz_net_names``), so ``--net`` here can never instantiate a
card-space ``BaseNet``.
"""
from regi_py.rl.adz.nets import get_adz_net, adz_net_names
from regi_py.rl.adz.explorer import (
    ADZNode,
    adz_simulate_node,
    ADZDirectStrategy,
    ADZExplorerStrategy,
)
from regi_py.rl.adz_training import (
    RecordingADZBruteStrategy,
    RecordingADZTeamStrategy,
    adz_infos_from_game,
)
from regi_py.rl.trainer_loop import Pipeline, Paradigm, run_trainer

ADZ_PARADIGM = Paradigm(
    node_cls=ADZNode,
    simulate_fn=adz_simulate_node,
    brute_recorder=RecordingADZBruteStrategy,
    team_recorder=RecordingADZTeamStrategy,
    infos_fn=adz_infos_from_game,
    direct_strat=ADZDirectStrategy,
    explorer_strat=ADZExplorerStrategy,
)

PIPELINE = Pipeline(
    prog="regi-adz-trainer",
    label="adz",
    net_default="adzmulti",
    net_choices=adz_net_names(),
    get_net=get_adz_net,
    paradigm=ADZ_PARADIGM,
)


if __name__ == "__main__":
    run_trainer(PIPELINE)
