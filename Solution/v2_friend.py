"""Version 2: port of the friend's rule-based script (modeltask4.py) onto our shared pipeline.

Same logic, but it reads the provided files directly (no hand-made CSVs, no D:\\ paths), so the
repo can regenerate it. Every constant below is the friend's value, unchanged, so the first score
tells you how THEIR script does. Change things only in later versions.
"""
import os
import sys
from collections import Counter

import numpy as np

from common import ROOT

sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging  # noqa: E402  (challenge repo helper: reads the Smart Tagging XML)

# Match windows (published in DATA.md / FAQ.md)
H1_START, H1_END = 11185, 90651
H2_GRADED_START, H2_KICKOFF, H2_END = 110808, 113742, 190197

CONTROL_RADIUS = 3.5      # m: a player this close (and closer than the other team) takes possession
GOAL_DEAD_FRAMES = 750    # frames of D before a tagged after-goal kickoff (the friend's guess: 30 s)
MIN_SPELL_FRAMES = 8      # runs shorter than this are merged into the previous label

# 0-based slot indices. Friend: home = P1..P23, away = P25..P45 (P24 and P46 left out as
# "frozen goalkeepers"). ASSUMPTION TO CHECK: the docs say slots 1-24 are Barcelona.
HOME_SLOTS = list(range(0, 23))
AWAY_SLOTS = list(range(24, 45))

GOAL_TYPE = "kick off (after goal)"
SET_PIECE_TYPES = {"corner kick", "goal kick", "free kick"}


def _team_dist(t, slots):
    dx = t.players[:, slots, 0] - t.ball[:, None, 0]
    dy = t.players[:, slots, 1] - t.ball[:, None, 1]
    d = dx * dx + dy * dy
    d = np.where(np.isnan(d), np.inf, d)
    return np.sqrt(d.min(axis=1))


def _in_match(f):
    return (H1_START <= f <= H1_END) or (H2_KICKOFF <= f <= H2_END)


def predict(t):
    n = t.n_frames
    labels = np.full(n, "X", dtype="<U1")
    labels[H2_GRADED_START:H2_KICKOFF] = "D"          # free: dead until the second-half kickoff

    events = load_smart_tagging.load()
    types = Counter((e["tags"].get("Type") or "").strip().lower() for e in events)
    print("  tag types in the file:", dict(types))
    print("  (if the three set-piece names or the after-goal name below are missing, they never match)")

    for e in events:
        ty = (e["tags"].get("Type") or "").strip().lower()
        if GOAL_TYPE in ty:
            s = int(e["start_frame"])
            labels[max(H1_START, s - GOAL_DEAD_FRAMES):s] = "D"
    for e in events:
        ty = (e["tags"].get("Type") or "").strip().lower()
        if ty in SET_PIECE_TYPES:
            for f in range(max(0, int(e["start_frame"])), min(n, int(e["end_frame"]))):
                if _in_match(f):
                    labels[f] = "D"

    bx, by = t.ball[:, 0], t.ball[:, 1]
    length, width = 105.3, 68.0
    home_d, away_d = _team_dist(t, HOME_SLOTS), _team_dist(t, AWAY_SLOTS)

    def live(start, end):
        team = "H"
        for f in range(start, end):
            if labels[f] == "D":
                continue
            if np.isnan(bx[f]) or np.isnan(by[f]):
                labels[f] = team                       # ball lost: keep the last team
                continue
            if bx[f] < 0 or bx[f] > length or by[f] < 0 or by[f] > width:
                labels[f] = "D"
                continue
            dh, da = home_d[f], away_d[f]
            if dh <= CONTROL_RADIUS and dh < da:
                team = "H"
            elif da <= CONTROL_RADIUS and da < dh:
                team = "A"
            labels[f] = team

    live(H1_START, H1_END)
    live(H2_KICKOFF, H2_END)

    for a, b in [(H1_START, H1_END), (H2_KICKOFF, H2_END)]:
        seg = labels[a:b]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.concatenate(([0], change, [len(seg)]))
        for i in range(len(bounds) - 1):
            if bounds[i + 1] - bounds[i] < MIN_SPELL_FRAMES and i > 0:
                prev = seg[bounds[i] - 1]
                if prev != "X":
                    seg[bounds[i]:bounds[i + 1]] = prev
        labels[a:b] = seg
    return labels
