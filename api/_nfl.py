"""Shared helpers for the NFL predictor API endpoints.

Reads the committed CSVs under the highest-numbered SeasonNN/ folder:
  - data/processed/predictions_<season>_wk<N>.csv  (model output, committed ahead of time)
  - data/raw/schedule_<season>.csv                 (refreshed each time get_week.py runs)

Actual results are joined from the schedule by game_id, so once a week's games
are played (and the schedule CSV is regenerated) the API grades the picks
automatically — no change to the committed prediction files.
"""
import glob
import math
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def latest_season_dir() -> Path:
    """Highest-numbered SeasonNN/ folder, so a new season is just a new folder."""
    seasons = [
        p for p in REPO_ROOT.glob("Season*")
        if p.is_dir() and re.fullmatch(r"Season\d+", p.name)
    ]
    if not seasons:
        raise RuntimeError(f"No SeasonNN/ folder found under {REPO_ROOT}")
    return max(seasons, key=lambda p: int(p.name.replace("Season", "")))


SEASON_DIR = latest_season_dir()
PROCESSED_DIR = SEASON_DIR / "data" / "processed"
RAW_DIR = SEASON_DIR / "data" / "raw"


def _clean(value):
    """JSON-safe scalar: NaN/NaT -> None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def week_files() -> dict:
    """{week_number: csv_path} for every committed predictions file."""
    out = {}
    for f in glob.glob(str(PROCESSED_DIR / "predictions_*_wk*.csv")):
        m = re.search(r"_wk(\d+)\.csv$", f)
        if m:
            out[int(m.group(1))] = f
    return dict(sorted(out.items()))


def season_year() -> int | None:
    for f in week_files().values():
        m = re.search(r"predictions_(\d{4})_wk", Path(f).name)
        if m:
            return int(m.group(1))
    return None


def load_schedule() -> pd.DataFrame:
    """The season schedule CSV (data/raw/schedule_<year>.csv), or empty frame."""
    year = season_year()
    candidates = []
    if year:
        candidates.append(RAW_DIR / f"schedule_{year}.csv")
    candidates += [
        Path(p) for p in glob.glob(str(RAW_DIR / "schedule_*.csv"))
        if re.search(r"schedule_\d{4}\.csv$", p)
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def _kickoff(row) -> str | None:
    day = _clean(row.get("gameday"))
    if not day:
        return None
    t = _clean(row.get("gametime"))
    return f"{day}T{t}" if t else str(day)


def grade_week(week: int, threshold: float = 0.5, schedule: pd.DataFrame | None = None) -> dict:
    """Return {week, games:[...], record:{...}} for one week, with actual
    results and per-pick correctness joined in where games are final."""
    files = week_files()
    if week not in files:
        return {"week": week, "games": [], "record": _empty_record()}

    df = pd.read_csv(files[week])
    if "week" in df.columns:
        df = df[df["week"] == week].copy()

    # Always derive the pick from home_win_prob + the requested threshold, so the
    # threshold control actually does something (the committed predicted_winner
    # column is fixed at 0.5 and kept only for the CSV / git record).
    df["predicted_winner"] = df.apply(
        lambda r: r["home_team"] if r["home_win_prob"] >= threshold else r["away_team"],
        axis=1,
    )
    df["predicted_win_prob"] = df.apply(
        lambda r: r["home_win_prob"] if r["predicted_winner"] == r["home_team"]
        else 1 - r["home_win_prob"],
        axis=1,
    )
    df["confidence"] = (df["home_win_prob"] - 0.5).abs() * 2

    if schedule is None:
        schedule = load_schedule()
    sched_by_id = {}
    if not schedule.empty and "game_id" in schedule.columns:
        sched_by_id = {r["game_id"]: r for _, r in schedule.iterrows()}

    games = []
    graded = correct = 0
    for _, row in df.iterrows():
        s = sched_by_id.get(row.get("game_id"), {})
        home_score = _clean(s.get("home_score")) if len(s) else None
        away_score = _clean(s.get("away_score")) if len(s) else None
        is_final = home_score is not None and away_score is not None

        actual_winner = None
        pick_correct = None
        if is_final and home_score != away_score:
            actual_winner = row["home_team"] if home_score > away_score else row["away_team"]
            pick_correct = bool(actual_winner == row["predicted_winner"])
            graded += 1
            correct += int(pick_correct)

        games.append({
            "game_id": _clean(row.get("game_id")),
            "week": int(week),
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "kickoff": _kickoff(s) if len(s) else None,
            "weekday": _clean(s.get("weekday")) if len(s) else None,
            "spread_home": _clean(row.get("spread_home")),
            "predicted_winner": row["predicted_winner"],
            "home_win_prob": round(float(row["home_win_prob"]), 4),
            "predicted_win_prob": round(float(row["predicted_win_prob"]), 4),
            "confidence": round(float(row["confidence"]), 4),
            "status": "final" if is_final else "scheduled",
            "home_score": int(home_score) if is_final else None,
            "away_score": int(away_score) if is_final else None,
            "actual_winner": actual_winner,
            "correct": pick_correct,
        })

    return {
        "week": int(week),
        "games": games,
        "record": _record(graded, correct),
    }


def _empty_record() -> dict:
    return {"graded": 0, "correct": 0, "accuracy": None}


def _record(graded: int, correct: int) -> dict:
    return {
        "graded": graded,
        "correct": correct,
        "accuracy": round(correct / graded, 4) if graded else None,
    }


def season_summary(threshold: float = 0.5) -> dict:
    """weeks list, which week to land on, previous week, and records."""
    files = week_files()
    weeks = list(files)
    schedule = load_schedule()

    by_week = {}
    overall_graded = overall_correct = 0
    current = None
    for w in weeks:
        gw = grade_week(w, threshold, schedule)
        n_final = sum(1 for g in gw["games"] if g["status"] == "final")
        by_week[str(w)] = {
            "games": len(gw["games"]),
            "final": n_final,
            "graded": gw["record"]["graded"],
            "correct": gw["record"]["correct"],
            "accuracy": gw["record"]["accuracy"],
        }
        overall_graded += gw["record"]["graded"]
        overall_correct += gw["record"]["correct"]
        # "current" = earliest week that still has an unplayed game
        if current is None and n_final < len(gw["games"]):
            current = w

    if current is None:
        current = weeks[-1] if weeks else None

    previous = None
    for w in weeks:
        if current is not None and w < current:
            previous = w

    return {
        "season": season_year(),
        "weeks": weeks,
        "current_week": current,
        "previous_week": previous,
        "records": {
            "overall": _record(overall_graded, overall_correct),
            "by_week": by_week,
        },
    }
