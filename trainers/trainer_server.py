"""Self-play trainer with a GPU inference server (AZ + ADZ).

Same pipelines as ``trainers/trainer.py`` but routes self-play predicts to one CUDA
inference process (rl.play_server). Runs self-play + brute only, no team games; use
``trainers/trainer.py`` for the CPU path / team games / quick tests.

  python -m trainers.trainer_server --net adzmulti --infer-device cuda ...
"""
from regi_py.rl.trainer_loop import run_trainer
from trainers.trainer import AZ, ADZ

if __name__ == "__main__":
    run_trainer([AZ, ADZ], infer_server=True)
