"""One-off: summarize a benchmark ZIP of j2df CSVs into per-team progress tallies.

The input ZIP holds one or more CSVs produced by ``j2df.py`` (event-log rows, one
row per game event).  For each CSV this writes a summary CSV to the output folder
with one row per team: how many games landed in each progress band and how many
were wins.  A game's progress is the max ``game.progress`` over its rows; a win is
progress 360 (all enemies cleared).  Each (game, team) is scored by its BEST
repetition (the ``sim`` with the highest final progress).

    python bench_summary.py <bench.zip> -o <out_folder> --run-id <id>
"""

import argparse
import io
import os
import zipfile

import pandas as pd

WIN_PROGRESS = 360
USECOLS = ["game", "team", "sim", "game.progress"]


def summarize_csv(fileobj, run_id):
    df = pd.read_csv(fileobj, usecols=USECOLS)

    # final progress of each played game = max progress over its event rows
    final = df.groupby(["game", "team", "sim"])["game.progress"].max()
    # best repetition per (game, team) = the sim with the highest final progress
    best = final.groupby(level=["game", "team"]).max().reset_index()

    rows = []
    for team, grp in best.groupby("team"):
        prog = grp["game.progress"]
        rows.append(
            {
                "team": team,
                "run_id": run_id,
                "progress_0_80": int(((prog >= 0) & (prog <= 80)).sum()),
                "progress_81_200": int(((prog >= 81) & (prog <= 200)).sum()),
                "progress_201_360": int(((prog >= 201) & (prog <= 360)).sum()),
                "wins": int((prog >= WIN_PROGRESS).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("team").reset_index(drop=True)


def out_name(member):
    # "2-p/stats.csv" -> "2-p-stats-summary.csv"
    stem = os.path.splitext(member)[0].strip("/").replace("/", "-")
    return f"{stem}-summary.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zip", help="benchmark ZIP of j2df CSVs")
    ap.add_argument("-o", "--output", required=True, help="output folder for summaries")
    ap.add_argument("--run-id", required=True, help="value for the summary 'run_id' column")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    with zipfile.ZipFile(args.zip) as z:
        members = [n for n in z.namelist() if n.endswith(".csv")]
        if not members:
            raise SystemExit(f"no CSV members found in {args.zip}")
        for member in members:
            print(f"summarizing {member}")
            with z.open(member) as f:
                summary = summarize_csv(io.TextIOWrapper(f, "utf-8"), args.run_id)
            dest = os.path.join(args.output, out_name(member))
            summary.to_csv(dest, index=False)
            print(f"  wrote {dest} ({len(summary)} teams)")


if __name__ == "__main__":
    main()
