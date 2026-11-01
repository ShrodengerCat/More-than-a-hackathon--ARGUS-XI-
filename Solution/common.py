"""Shared helpers for every solution version: load tracking (cached), write a submission, sanity-check it."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, asset_paths, source_frames  # noqa: E402  (challenge repo helper)

TEAM = "myteam"                      # <-- change to your real team name
CACHE_DIR = os.path.join(ROOT, "cache")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
N_SLOTS, N_HOME = 46, 24             # slots 0-23 = Barcelona (H), 24-45 = Athletic (A)
HOME, AWAY, NOBODY = 0, 1, -1


class Tracking:
    """players: (frames, 46, 3) = x, y (metres) and speed.  ball: (frames, 3).  NaN = not detected.
    Row i is video frame i (the file's frame_id minus 1), which is exactly the submission row."""
    def __init__(self, players, ball):
        self.players, self.ball = players, ball
        self.n_frames = len(ball)


def load_tracking(force=False):
    """Parse tactical_atd.txt once (takes a minute or two) and cache it as a .npz file."""
    path = os.path.join(CACHE_DIR, "tracking.npz")
    if os.path.exists(path) and not force:
        z = np.load(path)
        return Tracking(z["players"], z["ball"])

    raw_path = asset_paths()["atd_raw"]
    n = source_frames("tactical")
    length, width = GAMES[GAME]["pitch"]
    scale = np.array([length, width, 1.0], dtype=np.float32)   # x, y -> metres; speed untouched
    players = np.full((n, N_SLOTS, 3), np.nan, dtype=np.float32)
    ball = np.full((n, 3), np.nan, dtype=np.float32)

    print("Reading", os.path.basename(raw_path), "(one-off, then cached) ...")
    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            if not 0 <= i < n:
                continue
            p = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32).reshape(-1, 3)
            players[i, :min(len(p), N_SLOTS)] = p[:N_SLOTS] * scale
            ball[i] = np.array(parts[2].split(","), dtype=np.float32)[:3] * scale
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, players=players, ball=ball)
    return Tracking(players, ball)


def write_submission(name, labels):
    """labels: array of 'H'/'A'/'D'/'X', one per video frame. Writes evaluation/results/<TEAM>_<name>.csv"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "{}_{}.csv".format(TEAM, name))
    with open(path, "w") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    return path


def summarize(labels):
    """Quick sanity numbers: expect roughly 350-400 possessions of 10-20 s over a full match."""
    labels = np.asarray(labels)
    counts = {l: int(np.sum(labels == l)) for l in "HADX"}
    play = labels[(labels == "H") | (labels == "A")]
    # a possession = a run of the same H/A label with no D/X in between
    runs, cur, length = [], None, 0
    for l in labels:
        if l == cur and l in ("H", "A"):
            length += 1
        else:
            if cur in ("H", "A"):
                runs.append(length)
            cur, length = l, 1
    if cur in ("H", "A"):
        runs.append(length)
    mean_s = (np.mean(runs) / 25.0) if runs else 0.0
    print("  rows {}  H {}  A {}  D {}  X {}".format(len(labels), counts["H"], counts["A"], counts["D"], counts["X"]))
    print("  possessions: {}   mean length: {:.1f} s".format(len(runs), mean_s))
