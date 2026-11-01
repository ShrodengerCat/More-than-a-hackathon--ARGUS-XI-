"""Generate one submission from one version.

    python solution/run.py v1_baseline            # writes evaluation/results/<TEAM>_v1_baseline.csv
    python solution/run.py v1_baseline --score    # ...and then scores everything in that folder

To make a new version: copy v1_baseline.py to v2_something.py, change predict(), run it by name.
Every version file must define predict(tracking) -> array of 'H'/'A'/'D'/'X', one per frame.
"""
import argparse
import importlib
import subprocess
import sys

import common

parser = argparse.ArgumentParser()
parser.add_argument("version", help="file name without .py, e.g. v1_baseline")
parser.add_argument("--score", action="store_true", help="run the evaluation afterwards")
args = parser.parse_args()

tracking = common.load_tracking()
labels = importlib.import_module(args.version).predict(tracking)
assert len(labels) == tracking.n_frames, "wrong number of rows"
path = common.write_submission(args.version, labels)
print("wrote", path)
common.summarize(labels)

if args.score:
    subprocess.run([sys.executable, "evaluation/run_evaluation.py", "--bootstrap", "1000"], cwd=common.ROOT)
