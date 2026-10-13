"""
Weekly refresh: grade finished games and re-predict upcoming weeks.

    python Season26/scripts/weekly_update.py              # current week .. +7
    python Season26/scripts/weekly_update.py --ahead 4

1. Re-downloads data/raw/schedule_<season>.csv, which carries final scores, so
   the API grades every finished game automatically.
2. Re-runs scripts/predict_lab.py for the current NFL week (Tuesday through
   Monday Night Football, US Eastern) and the next `--ahead` weeks, skipping
   any week where a game has already been played — committed picks for a week
   in progress are never rewritten.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import subprocess
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.data import get_season_schedule, load_config, save_csv

LAST_REG_WEEK = 18


def today_eastern():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).date()
    except Exception:
        return datetime.now(timezone(timedelta(hours=-5))).date()


def main(args):
    cfg = load_config()
    season = int(cfg["season"])
    sched = get_season_schedule(season)
    save_csv(sched, f"{cfg['paths']['raw']}/schedule_{season}.csv")

    reg = sched[sched["game_type"] == "REG"].copy()
    reg["gameday"] = pd.to_datetime(reg["gameday"]).dt.date
    ends = reg.groupby("week")["gameday"].max()
    today = (datetime.strptime(args.today, "%Y-%m-%d").date() if args.today
             else today_eastern())
    current = next((int(w) for w, d in ends.items() if today <= d), None)

    played = reg.groupby("week")["home_score"].apply(lambda s: s.notna().any())
    final = reg.groupby("week")["home_score"].apply(lambda s: s.notna().all())
    done_weeks = [int(w) for w, f in final.items() if f]
    print(f"Schedule refreshed. Today {today} (ET); current week: {current}. "
          f"Fully final weeks: {done_weeks}", flush=True)

    if current is None:
        print("Regular season is over; nothing to predict.")
        return
    weeks = [w for w in range(current, min(current + args.ahead, LAST_REG_WEEK) + 1)
             if not played.get(w, False)]
    skipped = [w for w in range(current, min(current + args.ahead, LAST_REG_WEEK) + 1)
               if played.get(w, False)]
    if skipped:
        print(f"Not regenerating week(s) {skipped}: games already played.", flush=True)
    if not weeks:
        print("No unstarted weeks to predict.")
        return

    script = pathlib.Path(__file__).with_name("predict_lab.py")
    subprocess.run([sys.executable, str(script), "--weeks", *map(str, weeks)], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ahead", type=int, default=7, help="weeks past the current one to predict")
    ap.add_argument("--today", help="override today's date (YYYY-MM-DD), for testing")
    main(ap.parse_args())
