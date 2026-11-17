"""Generate tactical-video possession labels for More Than a Hack 2027."""

import argparse
import csv
import os
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FPS = 25
FRAME_COUNT = 199_319
PITCH = np.array([105.3, 68.0], dtype=np.float32)
# Published tactical video frames. The two graded endpoints are inclusive.
# Per official FAQ, second-half live play begins at video frame 113742;
# graded frames 110808..113741 are dead ball.
# This corrects one-frame kickoff alignment in experimental V3.
HALVES = ((11_185, 90_651, 11_185), (110_808, 190_197, 113_742))
TEAM_LABELS = {1: "H", -1: "A"}
PHASE_WEIGHTS = {
    "build up": 1.0,
    "progression": 1.0,
    "attacking transition": 1.0,
    "long ball": 0.65,
    "ball in final third": 0.45,
    "ball in the box": 0.45,
}
# Upper bounds for looking backward from a detected restart, in seconds.
RESTART_LOOKBACK = {"throw-in": 18, "goal kick": 35, "corner kick": 45, "free kick": 40}


@dataclass(frozen=True)
class Event:
    code: str
    kind: str
    start: int
    end: int
    team: int


def asset(root, suffix):
    name = f"BAR-ATH_tactical_{suffix}"
    for directory in (root / "tactical", root, root / "data" / "tactical"):
        if (path := directory / name).is_file():
            return path
    raise FileNotFoundError(f"Could not find {name} under {root}")


def tracking_metadata(path):
    root = ET.parse(path).getroot()
    home, away = list(root.iter("Team"))
    by_id = {home.get("id"): 1, away.get("id"): -1}
    by_name = {team.findtext("Name").strip().casefold(): by_id[team.get("id")]
               for team in (home, away)}
    players = sorted(root.iter("Player"), key=lambda player: int(player.findtext("ShirtNumber")))
    slots = [int(player.findtext("ShirtNumber")) for player in players]
    if slots != list(range(1, len(slots) + 1)):
        raise ValueError("ATD metadata must enumerate the tracking slots from 1")
    return np.array([by_id[player.get("teamId")] for player in players]), by_name


def load_tracking(path, slots):
    ball = np.empty((FRAME_COUNT, 2), dtype=np.float32)
    distances = np.empty((FRAME_COUNT, 2), dtype=np.float32)
    player_speed = np.empty(FRAME_COUNT, dtype=np.float32)
    batch_size = 2048
    batch = np.empty((batch_size, len(slots) + 1, 3), dtype=np.float32)

    def store_batch(start, end):
        rows = batch[:end - start]
        positions = rows[:, :, :2] * PITCH
        ball[start:end] = positions[:, -1]
        delta = positions[:, :-1] - positions[:, -1, None]
        radius = np.linalg.norm(delta, axis=2)
        radius[~np.isfinite(radius)] = np.inf
        for column, team in enumerate((1, -1)):
            distances[start:end, column] = radius[:, slots == team].min(axis=1)
        speeds = rows[:, :-1, 2]
        with warnings.catch_warnings():
            # A frame without any tracked players has unknown activity.
            warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
            player_speed[start:end] = np.nanmedian(speeds, axis=1)

    count = 0
    with path.open("rb") as handle:
        for count, line in enumerate(handle, start=1):
            frame_id, separator, values = line.partition(b":")
            if not separator or int(frame_id) != count or count > FRAME_COUNT:
                raise ValueError(f"Expected sequential tactical ATD frame ID {count} in {path}")
            parsed = np.fromstring(values.replace(b":", b",").replace(b";", b","),
                                   sep=",", dtype=np.float32)
            if parsed.size != batch.shape[1] * 3:
                raise ValueError(f"Unexpected ATD slot count on row {count} of {path}")
            # The one-based ATD ID maps to zero-based video frame count - 1.
            batch[(count - 1) % batch_size] = parsed.reshape(-1, 3)
            if count % batch_size == 0:
                store_batch(count - batch_size, count)
    if count != FRAME_COUNT:
        raise ValueError(f"Expected {FRAME_COUNT} tactical ATD rows, found {count} in {path}")
    if remainder := count % batch_size:
        store_batch(count - remainder, count)
    return ball, distances, player_speed


def load_events(path, teams):
    events = []
    for item in ET.parse(path).iter("instance"):
        tags = {label.findtext("group"): label.findtext("text", "").strip().casefold()
                for label in item.findall("label")}
        start = max(0, round(float(item.findtext("start")) * FPS))
        end = min(FRAME_COUNT, round(float(item.findtext("end")) * FPS))
        if start < end:
            events.append(Event(item.findtext("code", "").strip().casefold(),
                                tags.get("Type", ""), start, end, teams.get(tags.get("Team"), 0)))
    return sorted(events, key=lambda event: event.start)


def runs(mask):
    """Yield half-open intervals of true values."""
    changes = np.flatnonzero(np.diff(mask, prepend=False, append=False))
    return zip(changes[::2], changes[1::2])


def mean_at(values, start_offset, end_offset):
    """At each frame, average the finite samples in the given relative window."""
    finite = np.isfinite(values)
    total = np.r_[0.0, np.cumsum(np.where(finite, values, 0), dtype=np.float64)]
    count = np.r_[0, np.cumsum(finite)]
    frames = np.arange(len(values))
    left = np.clip(frames + start_offset, 0, len(values))
    right = np.clip(frames + end_offset, 0, len(values))
    n = count[right] - count[left]
    return np.divide(total[right] - total[left], n, out=np.full(len(values), np.nan), where=n > 0)


def bridge_ball(ball):
    """Interpolate short gaps whose endpoints can plausibly be one trajectory."""
    ball = ball.copy()
    for start, end in runs(~np.isfinite(ball).all(axis=1)):
        if start == 0 or end == len(ball) or end - start > FPS // 3:
            continue
        before, after = ball[start - 1], ball[end]
        if np.linalg.norm(after - before) <= (end - start + 1) * 30 / FPS:
            fraction = np.arange(1, end - start + 1)[:, None] / (end - start + 1)
            ball[start:end] = before + fraction * (after - before)
    return ball


def team_evidence(events, distance, edge, ball_speed):
    """Combine persistent ball contact with weak, bounded phase evidence."""
    prior = np.zeros_like(distance)
    for event in events:
        if event.team and event.code in PHASE_WEIGHTS:
            column = int(event.team == -1)
            interval = prior[event.start:event.end, column]
            np.maximum(interval, PHASE_WEIGHTS[event.code], out=interval)
    # Use the margin as well as the nearest distance. Dueling players give weak evidence.
    distance = np.minimum(distance, 20)
    home, away = distance.T
    margin = np.clip((away - home) / 1.5, -1, 1)
    proximity = np.clip((3.0 - np.minimum(home, away)) / 2, 0, 1)
    # A fast ball may be passing a defender without a controlled touch.
    flight_weight = np.clip(1.4 - np.nan_to_num(ball_speed, nan=8) / 15, 0.3, 1)
    contact = np.where(edge >= -0.4, margin * proximity * flight_weight, 0)
    return contact, contact + 0.30 * (prior[:, 0] - prior[:, 1])


def goal_departure_from_tracking(ball, phase_start, restart):
    """Find the end of a goal buildup using *unlabelled* ball observations.

    The supplied automatic GOALS tag may include minutes of buildup before
    the actual goal. A fixed phase_start + 30 seconds is not a goal timestamp.
    We look instead for a long ball-tracking gap that starts just after the
    last observed ball was in either goal area. This is conservative: an
    unrelated gap near midfield is not considered a scored goal.

    Nothing in this detector uses the public or private possession labels.
    """
    n = len(ball)
    lo = max(0, phase_start)
    hi = min(n, restart)
    if lo >= hi:
        return None
    valid = np.isfinite(ball).all(axis=1)
    for gap_a, gap_b in runs(~valid[lo:hi]):
        begin = lo + gap_a
        end = lo + gap_b
        if begin <= lo or end - begin < 3 * FPS or not valid[begin - 1]:
            continue
        last_pos = ball[begin - 1]
        x, y = float(last_pos[0]), float(last_pos[1])
        # Wide goal-mouth region; tracking is noisy and the ball may be several
        # metres outside the pitch as it disappears.
        near_goal = (min(abs(x), abs(PITCH[0] - x)) < 14
                     and abs(y - PITCH[1] / 2) < 16)
        if not near_goal:
            continue
        # A goal clip should not contain long stretches of normal visible play
        # *after* the inferred scoring moment. Reject isolated occlusions.
        remaining = valid[begin:hi]
        if remaining.size and remaining.mean() > 0.25:
            continue
        return begin
    return None


def detect_stoppages(ball, nearest, activity, events):
    """Estimate stoppage intervals and the team starting play when it is known."""
    count = len(ball)
    visible = np.isfinite(ball).all(axis=1)
    edge = np.min(np.minimum(ball, PITCH - ball), axis=1)
    # Use a short displacement to damp frame-to-frame tracker jitter.
    speed = np.full(count, np.nan)
    speed[2:-2] = np.linalg.norm(ball[4:] - ball[:-4], axis=1) * FPS / 4
    speed[~visible | (speed > 42)] = np.nan
    before = mean_at(speed, -25, -4)
    after = mean_at(speed, 0, 12)
    pace_before = mean_at(activity, -FPS, 0)
    pace_after = mean_at(activity, FPS // 2, 2 * FPS)
    visibility_before = mean_at(visible, -20, 0)
    live_motion = (edge > 1) & ((speed > 4) | ((speed > 2) & (nearest < 2)))
    steady_play = mean_at(live_motion, -6, 7) >= 0.55
    dead = np.zeros(count, dtype=bool)
    restart_team = np.zeros(count, dtype=np.int8)
    departures = np.zeros(count, dtype=bool)

    # Track runs with sustained observations beyond the line, bridging brief dropouts.
    outside = edge < -0.65
    for start, end in runs(~visible):
        if start and end < count and end - start <= FPS and outside[start - 1] and outside[end]:
            outside[start:end] = True
    for start, end in runs(outside):
        if end - start < round(0.4 * FPS):
            continue
        # Align to the crossing where it was tracked, with a buffer for calibration jitter.
        limit = max(0, start - FPS // 2)
        while start > limit and edge[start - 1] < 0:
            start -= 1
        departures[start] = True
        dead[start:end] = True

    # Missing ball can extend a stopped interval if pitch geometry or player motion supports it.
    for start, end in runs(~visible):
        if start == 0 or end == count or end - start < FPS:
            continue
        last, next_edge = edge[start - 1], edge[end]
        # Predict a nearby line crossing only for an outward trajectory; a camera losing
        # an aerial ball in midfield contributes no evidence of a stoppage.
        approach = edge[max(0, start - 6)] - last
        moving_out = 0 <= last < 3 and approach > 0.2 and last <= 2 * approach
        pace = gap_activity(activity[start:end])
        if (last < 0 or moving_out) and end - start <= 25 * FPS and (next_edge < 5 or pace < 1.4):
            crossing_delay = round(5 * last / approach) if moving_out else 0
            departures[start + crossing_delay] = True
            dead[start + crossing_delay:end] = True
        elif (2 * FPS <= end - start <= 25 * FPS
              and (pace < 0.95 or (pace < 1.35 and pace_before[start] - pace > 1.2))):
            dead[start:end] = True

    # Unlabelled pauses with a nearly static ball and a pronounced team-wide slowdown.
    still = (mean_at(speed, -12, 13) < 0.9) & (pace_before < 0.85) & visible
    for start, end in runs(still):
        if end - start >= 2 * FPS:
            dead[start:end] = True

    kickoffs = [event for event in events if event.kind == "kick off (after goal)"]
    goal_to_restart = []
    centre_distance = np.linalg.norm(ball - PITCH / 2, axis=1)
    # Goal clips include the buildup. Use the last shot/exit around the clip's pre-roll,
    # and find the corresponding centre restart rather than imposing a fixed celebration length.
    for goal in (event for event in events if event.code == "goals"):
        kickoff = next((event for event in kickoffs if event.start <= goal.end <= event.end), None)
        if kickoff is None:
            continue
        # Automatic goal clips are broad phases, not exact shot timestamps.
        # Locate the goal from the last ball observation near a goal line
        # before an extended disappearance, rather than an arbitrary 30 s
        # after the beginning of that broad phase.
        start = goal_departure_from_tracking(ball, goal.start, kickoff.start)
        if start is None:
            # Conservative fallback when no goal-line ball evidence exists.
            # Keeps this automatic tag from turning the full preceding phase
            # (possibly several minutes of play) into a stoppage.
            start = max(goal.start, kickoff.start - 30 * FPS)
        near_centre = np.flatnonzero((centre_distance[kickoff.start:kickoff.end] < 6)
                                    & (after[kickoff.start:kickoff.end] > 4))
        resume = kickoff.start + (near_centre[0] if near_centre.size else round(3.4 * FPS))
        goal_to_restart.append((start, resume, -goal.team))

    for event in events:
        if event.code != "set pieces" or event.kind not in RESTART_LOOKBACK:
            continue
        window = np.arange(event.start, min(event.end, event.start + 14 * FPS))
        if not window.size:
            continue
        # A restart should lead into moving play. Prefer an observed launch from the
        # appropriate pitch zone, then use the players' return to movement as corroboration.
        dx = np.minimum(ball[window, 0], PITCH[0] - ball[window, 0])
        dy = np.minimum(ball[window, 1], PITCH[1] - ball[window, 1])
        if event.kind == "throw-in":
            location = (dy >= -2) & (dy < 5)
        elif event.kind == "corner kick":
            location = (dx < 8) & (dy < 8) & (edge[window] >= -2)
        elif event.kind == "goal kick":
            location = (dx >= -1) & (dx < 10) & (abs(ball[window, 1] - PITCH[1] / 2) < 14)
        else:
            location = edge[window] >= 0

        pace_change = np.nan_to_num(pace_after[window] - pace_before[window])
        early_penalty = (window - event.start) / (14 * FPS)
        controlled_launch = (location & visible[window] & (after[window] >= 4)
                             & ((before[window] < 2.8) | ~np.isfinite(before[window])))
        movement_resume = (pace_before[window] < 1.25) & (pace_change > 0.8)
        reappears = (visibility_before[window] < 0.3) & (after[window] >= 4)
        rank = 2 * controlled_launch + 0.7 * reappears + np.clip(pace_change, 0, 2) - early_penalty
        possible = controlled_launch | (movement_resume & reappears) | (movement_resume & ~visible[window])
        if not possible.any():
            continue
        resume = window[np.argmax(np.where(possible, rank, -np.inf))]
        earliest = max(0, resume - RESTART_LOOKBACK[event.kind] * FPS)
        last_live = np.flatnonzero(steady_play[earliest:max(earliest, resume - FPS // 3)])
        start = earliest + last_live[-1] + 1 if last_live.size else earliest
        known_exits = np.flatnonzero(departures[start:resume])
        if known_exits.size:
            start += known_exits[-1]
        elif not last_live.size:
            # Without a visible end to the last spell, require local evidence of a pause.
            slow = np.flatnonzero(pace_before[start:resume] >= 1.25)
            if slow.size:
                start += slow[-1] + 1
        if resume - start >= FPS:
            dead[start:resume] = True
            restart_team[resume] = event.team
            # A live launch can be detected while the ball is still projected just outside.
            if event.kind == "throw-in" and controlled_launch[resume - event.start]:
                stop = min(count, resume + FPS)
                dead[resume:stop] = False

    # Goal restarts have reliable team context from the scorer tag.
    for start, resume, team in goal_to_restart:
        dead[start:resume] = True
        dead[resume:resume + FPS] = False
        restart_team[resume] = team

    # Join detector fragments separated by less than a half second of apparent play.
    for start, end in runs(~dead):
        if start and end < count and end - start < FPS // 2 and not restart_team[start:end].any():
            dead[start:end] = True
    return dead, restart_team, edge, speed


def gap_activity(values):
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size >= len(values) / 2 else np.inf


def possession_state(evidence, contact, dead, restart_team):
    labels = np.full(len(dead), "D", dtype="<U1")
    last_team = 0
    for start, end in runs(~dead):
        # A restart opens a fresh spell. Use upcoming early touches if no team was tagged.
        window_end = min(end, start + 2 * FPS)
        opening = evidence[start:window_end]
        if restart_team[start]:
            current = int(restart_team[start])
        elif opening.any():
            weights = np.linspace(1, 0.3, len(opening))
            current = 1 if np.dot(opening, weights) >= 0 else -1
        elif last_team:
            current = last_team
        else:
            support = evidence[start:end]
            hits = np.flatnonzero(support)
            if not hits.size:
                labels[start:end] = "X"
                continue
            current = 1 if support[hits[0]] > 0 else -1

        score = 0.0
        onset = start
        last_support = start
        spell_start = start
        for frame in range(start, end):
            opposing = -current * evidence[frame]
            if frame - last_support > FPS // 2:
                score = 0.0
            if opposing > 0:
                if score < 1:
                    onset = frame
                last_support = frame
            score = max(0.0, 0.99 * score + opposing)
            if score >= 17:
                # Confirm with sustained evidence, then place the boundary near its onset.
                boundary = max(spell_start, onset, frame - FPS)
                labels[spell_start:boundary] = TEAM_LABELS[current]
                current = -current
                spell_start = boundary
                score = 0.0
        labels[spell_start:end] = TEAM_LABELS[current]
        last_team = current

        # Absorb only short, weak A/B/A flickers inside the same uninterrupted live spell.
        spans = labels[start:end]
        bounds = np.r_[0, np.flatnonzero(spans[1:] != spans[:-1]) + 1, len(spans)]
        for left, right in zip(bounds[1:-2], bounds[2:-1]):
            sign = 1 if spans[left] == "H" else -1
            if (right - left < 2 * FPS and spans[left - 1] == spans[right]
                    and np.maximum(0, sign * contact[start + left:start + right]).sum() < 12):
                spans[left:right] = spans[left - 1]
    return labels


def predict(ball, distances, player_speed, events):
    labels = np.full(FRAME_COUNT, "X", dtype="<U1")
    # Process halves independently, so pregame and halftime observations cannot shape a spell.
    for first, last, kickoff in HALVES:
        end = last + 1
        local_events = [Event(event.code, event.kind, max(event.start, kickoff) - kickoff,
                              min(event.end, end) - kickoff, event.team)
                        for event in events if event.start < end and event.end > kickoff]
        half_ball = bridge_ball(ball[kickoff:end])
        half_distances = distances[kickoff:end]
        dead, restarts, edge, speed = detect_stoppages(
            half_ball, half_distances.min(axis=1), player_speed[kickoff:end], local_events)
        contact, evidence = team_evidence(local_events, half_distances, edge, speed)
        labels[first:kickoff] = "D"
        labels[kickoff:end] = possession_state(evidence, contact, dead, restarts)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.getenv("MTAH_DATA_DIR") or Path(__file__).resolve().parent))
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("model_task4_v3_submission.csv"))
    args = parser.parse_args()
    slots, names = tracking_metadata(asset(args.data_dir, "atd.xml"))
    ball, distances, activity = load_tracking(asset(args.data_dir, "atd.txt"), slots)
    events = load_events(asset(args.data_dir, "smart_tagging.xml"), names)
    labels = predict(ball, distances, activity, events)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("frame", "label"))
        writer.writerows(enumerate(labels))
    print(f"Wrote {len(labels):,} predictions to {args.output.resolve()}")
    print(" ".join(f"{label}={count:,}" for label, count in zip(*np.unique(labels, return_counts=True))))


if __name__ == "__main__":
    main()
