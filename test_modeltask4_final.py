"""Synthetic, unlabelled tests for modeltask4_v3.py (no competition data needed)."""
import importlib.util
import sys
from pathlib import Path
import numpy as np

FILE = Path(__file__).with_name('modeltask4_final.py')
spec = importlib.util.spec_from_file_location('modeltask4_v3', FILE)
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def check_goal_departure():
    ball = np.full((400, 2), np.nan, dtype=np.float32)
    ball[:40] = [50, 34]
    ball[40:65] = [103, 32]
    assert model.goal_departure_from_tracking(ball, 0, 350) == 65
    ball[:65] = [52, 34]
    assert model.goal_departure_from_tracking(ball, 0, 350) is None
    ball = np.full((400, 2), np.nan, dtype=np.float32)
    ball[:200] = [104, 33]
    # A brief occlusion isn't a scored goal.
    ball[150:180] = np.nan
    assert model.goal_departure_from_tracking(ball, 0, 350) == 200
    # If play resumes visibly, reject a short occlusion.
    ball[200:350] = [52, 34]
    assert model.goal_departure_from_tracking(ball, 0, 350) is None


def check_interpolation():
    ball = np.array([[5, 5], [np.nan, np.nan], [np.nan, np.nan], [8, 5]], dtype=np.float32)
    output = model.bridge_ball(ball)
    assert np.allclose(output[:, 0], [5, 6, 7, 8], equal_nan=False)


def check_kickoff():
    assert model.HALVES[1][2] == 113_742  # Official FAQ: frames 110808..113741 are D


def main():
    check_goal_departure()
    check_interpolation()
    check_kickoff()
    print('PASS: synthetic goal detection, interpolation, and kickoff alignment')


if __name__ == '__main__':
    main()
