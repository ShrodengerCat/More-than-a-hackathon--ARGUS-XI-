# ARGUS XI — More than a Hack 2027, Phase 1: Ball Possession

**Team:** ARGUS XI  
**Challenge:** Full-match tactical-video possession segmentation (H / A / D / X)  
**Organizers:** Metrica Sports, FC Barcelona, Mobile World Capital Barcelona

## Final submission implementation

**`modeltask4_final.py` is the final model used to generate the team's submission.**
The existing files in `Solution/` are exploratory versions and are **not** the final submission pipeline.

This deterministic algorithm combines automatic tactical player/ball tracking, nearest-player control evidence, automatic phase and restart tags, dead-ball heuristics, and temporal hysteresis. It processes the two halves independently and generates exactly 199,319 video-frame labels.

Read `ARGUS_XI.md` for algorithm details, inputs, limitations, and the no-training statement.

## Environment

Python 3.11+ recommended. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-argus.txt
```

The final model uses NumPy only. The organizer's evaluation scripts additionally require pandas.

## Input assets — private, never commit

Obtain these via the competition registration and place them at `./data/tactical/`:

- `BAR-ATH_tactical_atd.txt`
- `BAR-ATH_tactical_atd.xml`
- `BAR-ATH_tactical_smart_tagging.xml`

Alternatively point `--data-dir` to the local directory containing `tactical/`. The algorithm never requires match video or broadcast files.

## Reproduce the final CSV

```bash
python modeltask4_final.py --data-dir ./data --output ARGUS_XI.csv
python evaluation/validate_submission.py ARGUS_XI.csv
python test_modeltask4_final.py
```

Check the exact output's SHA-256 against:

`13530459b2c48dbaad70dadea0c5234c881c7f0a5d83d8559e01595cdc04d2fd`

This hash corresponds to the team's prepared CSV with the provided competition tracking/tagging bundle and Python/NumPy environment. For reproducibility, use the exact source and same assets. If hashes differ, investigate rather than manually changing labels.

## Submission

Submit `ARGUS_XI.csv`, this public repository URL, and `ARGUS_XI.md` through the official form emailed to registered participants. **Do not commit the submission CSV, restricted tracking arrays, match videos, or cache archives to GitHub.**

Official starter: https://github.com/metrica-sports/more-than-a-hack-2027
