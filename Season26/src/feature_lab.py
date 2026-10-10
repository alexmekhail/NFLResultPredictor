"""
Feature lab: build pre-game team features from past seasons and backtest which
combinations predict winners best.

Every feature for a game is computed only from games played *before* it, so a
backtest never peeks at the result it is predicting.

Data (cached under data/interim/, re-fetchable):
  - schedules (scores, betting line, rest, location) via nfl_data_py
  - nflverse team-game stats CSVs (turnovers, TDs, EPA, yards, sacks, penalties)
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TEAM_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_team/stats_team_week_{season}.csv"
)

# Schedules use the abbreviation at the time; stats use today's franchise.
RELOCATED = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# Candidate feature groups. Each game feature is (home team value - away team value)
# of a pre-game rolling stat, except spread / elo / rest which are per game.
FEATURE_GROUPS = {
    "spread":     ["spread_line"],
    "turnovers":  ["giveaways_diff", "takeaways_diff"],
    "touchdowns": ["tds_for_diff", "tds_against_diff"],
    "point_diff": ["pts_margin_diff"],
    "epa":        ["off_epa_diff", "def_epa_diff"],
    "yards":      ["yds_for_diff", "yds_against_diff"],
    "sacks":      ["sacks_made_diff", "sacks_taken_diff"],
    "penalties":  ["pen_yds_diff"],
    "win_pct":    ["winpct_diff"],
    "elo":        ["elo_diff"],
    "rest":       ["rest_diff"],
}
BASE_FEATURES = ["home_field"]  # 1 = true home game, 0 = neutral site

ROLLING_STATS = [
    "giveaways", "takeaways", "tds_for", "tds_against", "pts_margin",
    "off_epa", "def_epa", "yds_for", "yds_against", "sacks_made",
    "sacks_taken", "pen_yds",
]


# ── data ──────────────────────────────────────────────────────────────────────

def load_schedules(seasons, cache_dir) -> pd.DataFrame:
    import nfl_data_py as nfl

    seasons = sorted(int(s) for s in seasons)
    path = Path(cache_dir) / f"schedules_{seasons[0]}_{seasons[-1]}.csv"
    if path.exists():
        return pd.read_csv(path)
    df = nfl.import_schedules(seasons)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_team_stats(seasons, cache_dir) -> pd.DataFrame:
    frames = []
    for s in sorted(int(s) for s in seasons):
        path = Path(cache_dir) / f"stats_team_week_{s}.csv"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.read_csv(TEAM_STATS_URL.format(season=s)).to_csv(path, index=False)
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


# ── per team-game table ───────────────────────────────────────────────────────

def team_games(schedule: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """One row per team per completed game, with that game's raw stats."""
    sched = schedule[schedule["home_score"].notna() & schedule["away_score"].notna()].copy()
    for c in ("home_team", "away_team"):
        sched[c] = sched[c].replace(RELOCATED)

    st = stats.copy()
    st["giveaways"] = st["passing_interceptions"] + st["fumbles_lost_total"]
    st["tds"] = (st["passing_tds"] + st["rushing_tds"]
                 + st["special_teams_tds"].fillna(0) + st["def_tds"].fillna(0))
    st["epa"] = st["passing_epa"].fillna(0) + st["rushing_epa"].fillna(0)
    st["yds"] = st["passing_yards"] + st["rushing_yards"]
    keep = st[["game_id", "team", "giveaways", "tds", "epa", "yds",
               "def_sacks", "sacks_suffered", "penalty_yards"]]

    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        g = sched[["game_id", "season", "week", "gameday",
                   f"{side}_team", f"{other}_team", f"{side}_score", f"{other}_score"]].copy()
        g.columns = ["game_id", "season", "week", "gameday", "team", "opp", "pf", "pa"]
        rows.append(g)
    tg = pd.concat(rows, ignore_index=True)

    tg = tg.merge(keep, on=["game_id", "team"], how="left")
    opp = keep.rename(columns={"team": "opp", "giveaways": "opp_giveaways",
                               "tds": "opp_tds", "epa": "opp_epa", "yds": "opp_yds"})
    tg = tg.merge(opp[["game_id", "opp", "opp_giveaways", "opp_tds", "opp_epa", "opp_yds"]],
                  on=["game_id", "opp"], how="left")

    tg["win"] = np.where(tg["pf"] > tg["pa"], 1.0, np.where(tg["pf"] < tg["pa"], 0.0, 0.5))
    tg["takeaways"] = tg["opp_giveaways"]
    tg["tds_for"] = tg["tds"]
    tg["tds_against"] = tg["opp_tds"]
    tg["pts_margin"] = tg["pf"] - tg["pa"]
    tg["off_epa"] = tg["epa"]
    tg["def_epa"] = tg["opp_epa"]          # EPA allowed (lower is better)
    tg["yds_for"] = tg["yds"]
    tg["yds_against"] = tg["opp_yds"]
    tg["sacks_made"] = tg["def_sacks"]
    tg["sacks_taken"] = tg["sacks_suffered"]
    tg["pen_yds"] = tg["penalty_yards"]
    return tg.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def pregame_team_features(tg: pd.DataFrame, window: int = 8, win_seasons: int = 3) -> pd.DataFrame:
    """For each team-game, stats known *before* kickoff.

    - roll_<stat>: mean over the team's previous `window` games (spans seasons,
      so week 1 uses last season's tail).
    - winpct: win % over the previous `win_seasons` full seasons plus this
      season's games so far.
    """
    tg = tg.copy()
    by_team = tg.groupby("team", sort=False)
    for s in ROLLING_STATS:
        tg[f"roll_{s}"] = by_team[s].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean())

    # Season totals per team, then the trailing K-season sum before this season.
    season_tot = tg.groupby(["team", "season"])["win"].agg(["sum", "count"]).reset_index()
    prior = []
    for _, r in season_tot.iterrows():
        m = (season_tot["team"] == r["team"]) & season_tot["season"].between(
            r["season"] - win_seasons, r["season"] - 1)
        prior.append((season_tot.loc[m, "sum"].sum(), season_tot.loc[m, "count"].sum()))
    season_tot["prior_w"], season_tot["prior_g"] = zip(*prior)
    tg = tg.merge(season_tot[["team", "season", "prior_w", "prior_g"]], on=["team", "season"])

    by_ts = tg.sort_values(["gameday", "game_id"]).groupby(["team", "season"], sort=False)
    tg["cur_w"] = by_ts["win"].transform(lambda x: x.shift(1).fillna(0).cumsum())
    tg["cur_g"] = by_ts["win"].transform(lambda x: x.shift(1).notna().cumsum())
    games = tg["prior_g"] + tg["cur_g"]
    tg["winpct"] = np.where(games > 0, (tg["prior_w"] + tg["cur_w"]) / games.replace(0, 1), 0.5)
    return tg


def elo_ratings(schedule: pd.DataFrame, k: float = 20.0, hfa: float = 48.0,
                revert: float = 1 / 3) -> pd.DataFrame:
    """Pre-game Elo for every completed game (538-style margin multiplier)."""
    g = schedule[schedule["home_score"].notna()].sort_values(["gameday", "game_id"])
    elo, last_season, out = {}, {}, []
    for r in g.itertuples():
        h, a = RELOCATED.get(r.home_team, r.home_team), RELOCATED.get(r.away_team, r.away_team)
        for t in (h, a):
            if t not in elo:
                elo[t] = 1500.0
            elif last_season.get(t) != r.season:
                elo[t] = elo[t] * (1 - revert) + 1505 * revert
            last_season[t] = r.season
        home_adv = hfa if r.location != "Neutral" else 0.0
        diff = elo[h] + home_adv - elo[a]
        out.append((r.game_id, elo[h] - elo[a]))
        exp_h = 1 / (1 + 10 ** (-diff / 400))
        margin = r.home_score - r.away_score
        res_h = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        mult = np.log(abs(margin) + 1) * 2.2 / ((diff if margin > 0 else -diff) * 0.001 + 2.2)
        delta = k * mult * (res_h - exp_h)
        elo[h] += delta
        elo[a] -= delta
    return pd.DataFrame(out, columns=["game_id", "elo_diff"])


# ── game matrix ───────────────────────────────────────────────────────────────

def game_matrix(schedule: pd.DataFrame, stats: pd.DataFrame,
                window: int = 8, win_seasons: int = 3) -> pd.DataFrame:
    """One row per completed, non-tie game: label + every candidate feature."""
    tg = pregame_team_features(team_games(schedule, stats), window, win_seasons)
    feat_cols = [f"roll_{s}" for s in ROLLING_STATS] + ["winpct"]
    side = tg[["game_id", "team"] + feat_cols]

    g = schedule[schedule["home_score"].notna() & schedule["away_score"].notna()].copy()
    g = g[g["home_score"] != g["away_score"]]
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(RELOCATED)
    g = g.merge(side.add_prefix("h_"), left_on=["game_id", "home_team"],
                right_on=["h_game_id", "h_team"])
    g = g.merge(side.add_prefix("a_"), left_on=["game_id", "away_team"],
                right_on=["a_game_id", "a_team"])

    for s in ROLLING_STATS:
        g[f"{s}_diff"] = g[f"h_roll_{s}"] - g[f"a_roll_{s}"]
    g["winpct_diff"] = g["h_winpct"] - g["a_winpct"]
    g["rest_diff"] = g["home_rest"] - g["away_rest"]
    g["home_field"] = (g["location"] != "Neutral").astype(float)
    g = g.merge(elo_ratings(schedule), on="game_id", how="left")
    g["home_win"] = (g["home_score"] > g["away_score"]).astype(int)

    cols = (["game_id", "season", "week", "game_type", "home_team", "away_team", "home_win"]
            + BASE_FEATURES + sorted({c for cs in FEATURE_GROUPS.values() for c in cs}))
    return g[cols].reset_index(drop=True)


# ── backtest ──────────────────────────────────────────────────────────────────

def walk_forward(games: pd.DataFrame, features, test_seasons, train_window: int = 3,
                 C: float = 1.0, return_preds: bool = False) -> dict:
    """For each test season S: fit on seasons [S-train_window, S-1], predict S.

    Returns pooled accuracy / log loss / Brier and per-season accuracy.
    """
    cols = BASE_FEATURES + list(features)
    probs, ys, ids, per_season = [], [], [], {}
    for s in test_seasons:
        tr = games[games["season"].between(s - train_window, s - 1)].dropna(subset=cols)
        te = games[games["season"] == s].dropna(subset=cols)
        if tr.empty or te.empty:
            continue
        model = Pipeline([("scale", StandardScaler()),
                          ("lr", LogisticRegression(C=C, max_iter=1000))])
        model.fit(tr[cols].values, tr["home_win"].values)
        p = model.predict_proba(te[cols].values)[:, 1]
        per_season[s] = float(((p >= 0.5) == te["home_win"].values).mean())
        probs.append(p)
        ys.append(te["home_win"].values)
        ids.append(te["game_id"].values)
    p, y = np.concatenate(probs), np.concatenate(ys)
    out = {
        "accuracy": float(((p >= 0.5) == y).mean()),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "games": int(len(y)),
        "per_season": per_season,
    }
    if return_preds:
        out["preds"] = pd.DataFrame({"game_id": np.concatenate(ids), "prob_home": p, "home_win": y})
    return out


def all_combinations(groups=None, max_size=None):
    groups = list(groups or FEATURE_GROUPS)
    max_size = max_size or len(groups)
    for k in range(1, max_size + 1):
        yield from combinations(groups, k)


def group_columns(combo):
    return [c for g in combo for c in FEATURE_GROUPS[g]]
