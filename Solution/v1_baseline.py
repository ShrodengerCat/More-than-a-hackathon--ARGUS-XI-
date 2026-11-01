"""Version 1: re-implementation of the provided baseline (nearest player to the ball + majority vote).
Should score about the same as baseline_smooth50 (0.2433). If it does, your setup is correct."""
import numpy as np

from common import HOME, AWAY, NOBODY, N_HOME

WINDOW = 50   # half-width in frames (25 frames = 1 second)


def nearest_team(t):
    """Per frame: team of the player nearest the ball, NOBODY where the ball is missing."""
    dx = t.players[:, :, 0] - t.ball[:, None, 0]
    dy = t.players[:, :, 1] - t.ball[:, None, 1]
    d = dx * dx + dy * dy
    d = np.where(np.isnan(d), np.inf, d)
    nearest = d.argmin(axis=1)
    codes = np.where(nearest < N_HOME, HOME, AWAY).astype(np.int8)
    codes[np.isnan(t.ball[:, 0])] = NOBODY
    codes[np.isinf(d.min(axis=1))] = NOBODY
    return codes


def smooth(codes, window):
    """Majority vote over +/- window frames; D where nobody has the ball in that window."""
    home = np.concatenate(([0], np.cumsum(codes == HOME)))
    away = np.concatenate(([0], np.cumsum(codes == AWAY)))
    idx = np.arange(len(codes))
    lo, hi = np.maximum(0, idx - window), np.minimum(len(codes), idx + window + 1)
    h, a = home[hi] - home[lo], away[hi] - away[lo]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (h + a) > 0
    labels[seen] = np.where(h[seen] >= a[seen], "H", "A")
    return labels


def predict(tracking):
    return smooth(nearest_team(tracking), WINDOW)
