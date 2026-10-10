"""
Predict upcoming weeks with the feature-lab model.

    python Season26/scripts/predict_lab.py --weeks 6 7 8 9

Follows the same rules as the backtest (scripts/feature_search.py):
  - Only this season's played games and the 3 seasons before it are used.
  - The logistic regression trains on those 3 prior seasons' games.
  - Team features for every upcoming game are each team's values *as of now*.
  - Games with no betting line posted yet get a spread projected from this
    season's posted lines (src.features.project_spreads); spread_source says which.

Writes data/processed/predictions_<season>_wk<N>.csv in the dashboard's format.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_config
from src.feature_lab import (BASE_FEATURES, LOOKBACK_SEASONS, QB_INJURY_GROUPS, game_matrix,
                             group_columns, load_schedules, load_team_stats, upcoming_matrix)
from src.features import project_spreads

# Chosen from models/reports/feature_search*.md — see README "Feature lab".
# spread + turnovers + QB was the only combination that beat the spread on both
# validation (2016-19) and holdout (2020-25) seasons.
DEFAULT_GROUPS = ["spread", "turnovers", "qb_quality", "qb_vs_team", "qb_change"]
QB_GROUPS = {"qb_change", "qb_experience", "qb_quality", "qb_vs_team"}


def refresh_current_season(cache: pathlib.Path, season: int):
    """Drop cached files for the in-progress season so they're re-downloaded."""
    for f in cache.glob(f"*_{season}.csv"):
        f.unlink()


def main(args):
    cfg = load_config()
    season = int(cfg["season"])
    paths = cfg["paths"]
    cache = pathlib.Path(paths["interim"]) / "feature_lab"
    refresh_current_season(cache, season)

    seasons = list(range(season - LOOKBACK_SEASONS, season + 1))
    sched = load_schedules(seasons, cache)
    stats = load_team_stats(seasons, cache)
    sched = sched[sched["season"].isin(seasons)]

    cols = BASE_FEATURES + group_columns(args.groups)
    unsupported = set(args.groups) & (set(QB_INJURY_GROUPS) - QB_GROUPS)
    if unsupported:
        raise SystemExit(f"{sorted(unsupported)}: no injury report exists for future weeks")
    use_qb = bool(set(args.groups) & QB_GROUPS)
    if use_qb:
        from src.qb_injury_features import extra_game_features, upcoming_qb_features

    # Train on the 3 prior seasons, features built from the same 4-season slice.
    games = game_matrix(sched, stats, window=args.window, win_seasons=args.win_seasons)
    if use_qb:
        games = games.merge(extra_game_features(sched, cache, seasons), on="game_id", how="left")
    train = games[games["season"] < season].dropna(subset=cols)
    model = Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    model.fit(train[cols].values, train["home_win"].values)

    artifacts = pathlib.Path(paths["artifacts"])
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts / "lab_model.pkl")
    (artifacts / "lab_model.json").write_text(json.dumps({
        "groups": args.groups, "columns": cols, "season": season,
        "train_seasons": sorted(train["season"].unique().tolist()), "n_train": int(len(train)),
        "window": args.window, "win_seasons": args.win_seasons,
    }, indent=2))
    print(f"Trained on {len(train)} games ({seasons[0]}-{season - 1}) with {', '.join(args.groups)}")

    cur = sched[sched["season"] == season]
    for week in args.weeks:
        # Project lines only from this season's posted lines before this week.
        cur_w = project_spreads(cur, before_week=week)
        up = cur_w[(cur_w["week"] == week)]
        if up.empty:
            print(f"Week {week}: no games")
            continue
        X = upcoming_matrix(sched, stats, up, window=args.window, win_seasons=args.win_seasons)
        if use_qb:
            pending = cur_w[(cur_w["week"] <= week) & cur_w["home_score"].isna()]
            X = X.merge(upcoming_qb_features(sched, pending, cache, seasons), on="game_id", how="left")
        missing = X[cols].isna().any(axis=1)
        if missing.any():
            print(f"Week {week}: skipping {int(missing.sum())} games with missing features "
                  f"({', '.join(X.loc[missing, 'game_id'])})")
            X = X[~missing]
        p = model.predict_proba(X[cols].values)[:, 1]

        out = pd.DataFrame({
            "game_id": X["game_id"],
            "week": week,
            "home_team": up.set_index("game_id").loc[X["game_id"], "home_team"].values,
            "away_team": up.set_index("game_id").loc[X["game_id"], "away_team"].values,
            "spread_home": -X["spread_line"].values,
            "spread_source": X["spread_source"].values,
            "home_win_prob": p,
        })
        out["predicted_winner"] = out["home_team"].where(out["home_win_prob"] >= 0.5, out["away_team"])
        out["confidence"] = (out["home_win_prob"] - 0.5).abs() * 2
        out["model"] = "lab:" + "+".join(args.groups)
        if use_qb:
            # True where no starter was announced yet and last week's starter was assumed.
            assumed = X.set_index("game_id")
            out["qb_assumed"] = (assumed.loc[out["game_id"], "home_qb_assumed"].values
                                 | assumed.loc[out["game_id"], "away_qb_assumed"].values)
        out = out.sort_values("confidence", ascending=False)

        path = pathlib.Path(paths["processed"]) / f"predictions_{season}_wk{week}.csv"
        out.to_csv(path, index=False)
        n_proj = int((out["spread_source"] == "projected").sum())
        n_qb = int(out["qb_assumed"].sum()) if "qb_assumed" in out else 0
        notes = ([f"{n_proj}/{len(out)} projected spreads"] if n_proj else []) + \
                ([f"{n_qb}/{len(out)} assumed QBs"] if n_qb else [])
        print(f"\nWeek {week} -> {path.name}" + (f"  ({', '.join(notes)})" if notes else ""))
        fav = out["home_team"].where(out["spread_home"] < 0, out["away_team"])
        view = out[["away_team", "home_team", "spread_home", "home_win_prob", "predicted_winner"]].copy()
        view["vs_line"] = ["" if p == f or s == 0 else "UPSET PICK"
                           for p, f, s in zip(out["predicted_winner"], fav, out["spread_home"])]
        view["home_win_prob"] = view["home_win_prob"].round(3)
        print(view.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, nargs="+", required=True)
    ap.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS,
                    help="feature groups from src/feature_lab.py FEATURE_GROUPS")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--win-seasons", type=int, default=3)
    main(ap.parse_args())
