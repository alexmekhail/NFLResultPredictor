"""
Quarterback and injury features for the feature lab.

All values are known before kickoff:
  - QB features use the game's announced starter (schedule `*_qb_id`) and that
    QB's stats from *earlier* games only.
  - Injury load uses the final injury report for the week (Out / Doubtful)
    weighted by each player's snap share in his games *before* this one.

Data (nflverse releases, cached under data/interim/feature_lab/):
  stats_player_week_<yr>.csv  (QB rows only are kept)   2002+
  injuries_<yr>.csv                                     2009+
  snap_counts_<yr>.csv                                  2013+
  players.csv  (gsis_id <-> pfr_id)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_lab import RELOCATED

RELEASE = "https://github.com/nflverse/nflverse-data/releases/download/{tag}/{name}"

# EPA/play a QB with no track record is assumed to have (roughly replacement
# level), and how many plays of his own it takes to move halfway off it.
QB_PRIOR_EPA = -0.10
QB_PRIOR_PLAYS = 150
QB_HISTORY_GAMES = 16      # QB rating = his last 16 games played, any team
TEAM_QB_BASELINE_GAMES = 8
SNAP_HISTORY_GAMES = 4     # a player's weight = his snap share over his last 4 games

def _cached(tag, name, cache_dir, usecols=None, keep=None) -> pd.DataFrame:
    path = Path(cache_dir) / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(RELEASE.format(tag=tag, name=name), usecols=usecols, low_memory=False)
        if keep is not None:
            df = keep(df)
        df.to_csv(path, index=False)
    return pd.read_csv(path, low_memory=False)


def _order(season, week):
    """Sortable game-time key (postseason weeks are 19-22)."""
    return season * 100 + week


# ── QB ────────────────────────────────────────────────────────────────────────

def load_qb_stats(seasons, cache_dir) -> pd.DataFrame:
    cols = ["player_id", "position", "season", "week", "attempts", "sacks_suffered",
            "carries", "passing_epa", "rushing_epa"]
    frames = [
        _cached("stats_player", f"stats_player_week_{s}.csv", cache_dir, usecols=cols,
                keep=lambda d: d[d["position"] == "QB"])
        for s in seasons
    ]
    return pd.concat(frames, ignore_index=True)


def qb_ratings(qb_stats: pd.DataFrame) -> pd.DataFrame:
    """Per QB per game played: his shrunk EPA/play over the previous
    QB_HISTORY_GAMES games *before* that game (`rating_before`)."""
    q = qb_stats.copy()
    q["plays"] = q["attempts"].fillna(0) + q["sacks_suffered"].fillna(0) + q["carries"].fillna(0)
    q["epa"] = q["passing_epa"].fillna(0) + q["rushing_epa"].fillna(0)
    q = q[q["plays"] > 0]
    q["key"] = _order(q["season"], q["week"])
    q = q.sort_values(["player_id", "key"])
    g = q.groupby("player_id", sort=False)
    roll = lambda s: s.shift(1).rolling(QB_HISTORY_GAMES, min_periods=1).sum()
    q["epa_hist"] = g["epa"].transform(roll).fillna(0)
    q["plays_hist"] = g["plays"].transform(roll).fillna(0)
    # Rating that includes this game, for looking up "latest rating before game X".
    q["epa_incl"] = g["epa"].transform(lambda s: s.rolling(QB_HISTORY_GAMES, min_periods=1).sum())
    q["plays_incl"] = g["plays"].transform(lambda s: s.rolling(QB_HISTORY_GAMES, min_periods=1).sum())
    q["rating_after"] = ((q["epa_incl"] + QB_PRIOR_EPA * QB_PRIOR_PLAYS)
                         / (q["plays_incl"] + QB_PRIOR_PLAYS))
    return q[["player_id", "key", "rating_after"]]


def qb_features(schedule: pd.DataFrame, qb_stats: pd.DataFrame) -> pd.DataFrame:
    """Per team-game: qb_change, qb_starts, qb_epa, qb_vs_team."""
    s = schedule.copy()
    rows = []
    for side in ("home", "away"):
        r = s[["game_id", "season", "week", "gameday", f"{side}_team", f"{side}_qb_id"]].copy()
        r.columns = ["game_id", "season", "week", "gameday", "team", "qb_id"]
        rows.append(r)
    tg = pd.concat(rows, ignore_index=True)
    tg["team"] = tg["team"].replace(RELOCATED)
    tg["key"] = _order(tg["season"], tg["week"])
    tg = tg.sort_values(["key", "gameday", "game_id"]).reset_index(drop=True)

    # Change of starter vs the team's previous game.
    prev = tg.groupby("team")["qb_id"].shift(1)
    tg["qb_change"] = ((prev.notna()) & (prev != tg["qb_id"])).astype(float)

    # Career starts before this game (any team).
    tg["qb_starts"] = np.log1p(tg.groupby("qb_id").cumcount())

    # QB's rating from his games strictly before this one.
    ratings = qb_ratings(qb_stats).rename(columns={"player_id": "qb_id"})
    tg = tg.sort_values("key")
    ratings = ratings.sort_values("key")
    tg = pd.merge_asof(tg, ratings, on="key", by="qb_id", allow_exact_matches=False)
    tg["qb_epa"] = tg["rating_after"].fillna(QB_PRIOR_EPA)

    # Team baseline: average rating of whoever started its previous N games.
    tg = tg.sort_values(["key", "gameday", "game_id"])
    tg["team_qb_base"] = tg.groupby("team")["qb_epa"].transform(
        lambda x: x.shift(1).rolling(TEAM_QB_BASELINE_GAMES, min_periods=1).mean())
    tg["qb_vs_team"] = (tg["qb_epa"] - tg["team_qb_base"]).fillna(0)
    return tg[["game_id", "team", "qb_change", "qb_starts", "qb_epa", "qb_vs_team"]]


# ── injuries ─────────────────────────────────────────────────────────────────

def load_injuries(seasons, cache_dir) -> pd.DataFrame:
    cols = ["season", "week", "team", "gsis_id", "position", "report_status"]
    frames = [_cached("injuries", f"injuries_{s}.csv", cache_dir, usecols=cols)
              for s in seasons if s >= 2009]
    inj = pd.concat(frames, ignore_index=True)
    inj["team"] = inj["team"].replace(RELOCATED)
    return inj[inj["report_status"].isin(["Out", "Doubtful"])]


def load_snaps(seasons, cache_dir) -> pd.DataFrame:
    cols = ["season", "week", "pfr_player_id", "offense_pct", "defense_pct"]
    frames = [_cached("snap_counts", f"snap_counts_{s}.csv", cache_dir, usecols=cols)
              for s in seasons if s >= 2013]
    return pd.concat(frames, ignore_index=True)


def injury_features(schedule: pd.DataFrame, cache_dir, seasons) -> pd.DataFrame:
    """Per team-game: inj_count (Out/Doubtful non-QBs) and snap-weighted
    inj_off / inj_def load (only where snap data exists, 2013+)."""
    inj = load_injuries(seasons, cache_dir)
    inj = inj[inj["position"] != "QB"]          # QBs are covered by the QB features
    inj["key"] = _order(inj["season"], inj["week"])

    players = _cached("players", "players.csv", cache_dir, usecols=["gsis_id", "pfr_id"])
    snaps = load_snaps(seasons, cache_dir).rename(columns={"pfr_player_id": "pfr_id"})
    snaps["key"] = _order(snaps["season"], snaps["week"])
    snaps = snaps.sort_values(["pfr_id", "key"])
    g = snaps.groupby("pfr_id", sort=False)
    snaps["off_share"] = g["offense_pct"].transform(
        lambda s: s.rolling(SNAP_HISTORY_GAMES, min_periods=1).mean())
    snaps["def_share"] = g["defense_pct"].transform(
        lambda s: s.rolling(SNAP_HISTORY_GAMES, min_periods=1).mean())

    inj = inj.merge(players.dropna().drop_duplicates("gsis_id"), on="gsis_id", how="left")
    inj = pd.merge_asof(inj.sort_values("key"), snaps[["pfr_id", "key", "off_share", "def_share"]]
                        .sort_values("key"), on="key", by="pfr_id", allow_exact_matches=False)
    is_off = inj["off_share"].fillna(0) >= inj["def_share"].fillna(0)
    inj["off_load"] = np.where(is_off, inj["off_share"].fillna(0), 0.0)
    inj["def_load"] = np.where(~is_off, inj["def_share"].fillna(0), 0.0)

    agg = inj.groupby(["season", "week", "team"]).agg(
        inj_count=("gsis_id", "count"), inj_off=("off_load", "sum"), inj_def=("def_load", "sum")
    ).reset_index()

    rows = []
    for side in ("home", "away"):
        r = schedule[["game_id", "season", "week", f"{side}_team"]].copy()
        r.columns = ["game_id", "season", "week", "team"]
        rows.append(r)
    tg = pd.concat(rows, ignore_index=True)
    tg["team"] = tg["team"].replace(RELOCATED)
    tg = tg.merge(agg, on=["season", "week", "team"], how="left")
    has_reports = tg["season"] >= 2009
    has_snaps = tg["season"] >= 2013
    tg["inj_count"] = np.where(has_reports, tg["inj_count"].fillna(0), np.nan)
    for c in ("inj_off", "inj_def"):
        tg[c] = np.where(has_snaps, tg[c].fillna(0), np.nan)
    return tg[["game_id", "team", "inj_count", "inj_off", "inj_def"]]


# ── per game ─────────────────────────────────────────────────────────────────

def extra_game_features(schedule: pd.DataFrame, cache_dir, seasons) -> pd.DataFrame:
    """game_id + home-minus-away diffs for every QB and injury feature."""
    sched = schedule[schedule["home_score"].notna()].copy()
    team = qb_features(sched, load_qb_stats(seasons, cache_dir)).merge(
        injury_features(sched, cache_dir, seasons), on=["game_id", "team"], how="left")
    g = sched[["game_id", "home_team", "away_team"]].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(RELOCATED)
    g = g.merge(team.add_prefix("h_"), left_on=["game_id", "home_team"], right_on=["h_game_id", "h_team"])
    g = g.merge(team.add_prefix("a_"), left_on=["game_id", "away_team"], right_on=["a_game_id", "a_team"])
    out = g[["game_id"]].copy()
    for c in ("qb_change", "qb_starts", "qb_epa", "qb_vs_team", "inj_count", "inj_off", "inj_def"):
        out[f"{c}_diff"] = g[f"h_{c}"] - g[f"a_{c}"]
    return out


def upcoming_qb_features(schedule: pd.DataFrame, pending: pd.DataFrame, cache_dir, seasons) -> pd.DataFrame:
    """QB diffs for unplayed games in `pending` (every unplayed game up to the
    target week, so "QB change" compares against the game before it).

    Starters come from the schedule's announced `*_qb_id`; where none is listed
    yet, the team's most recent starter is assumed. Ratings use only games
    already played.
    """
    done = schedule[schedule["home_score"].notna()].sort_values(["gameday", "game_id"])
    last_qb = {}
    for r in done.itertuples():
        last_qb[RELOCATED.get(r.home_team, r.home_team)] = r.home_qb_id
        last_qb[RELOCATED.get(r.away_team, r.away_team)] = r.away_qb_id

    up = pending.copy()
    for side in ("home", "away"):
        team = up[f"{side}_team"].replace(RELOCATED)
        up[f"{side}_qb_assumed"] = up[f"{side}_qb_id"].isna()
        up[f"{side}_qb_id"] = up[f"{side}_qb_id"].fillna(team.map(last_qb))

    team = qb_features(pd.concat([done, up], ignore_index=True), load_qb_stats(seasons, cache_dir))
    g = up[["game_id", "home_team", "away_team", "home_qb_assumed", "away_qb_assumed"]].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(RELOCATED)
    g = g.merge(team.add_prefix("h_"), left_on=["game_id", "home_team"], right_on=["h_game_id", "h_team"])
    g = g.merge(team.add_prefix("a_"), left_on=["game_id", "away_team"], right_on=["a_game_id", "a_team"])
    out = g[["game_id", "home_qb_assumed", "away_qb_assumed"]].copy()
    for c in ("qb_change", "qb_starts", "qb_epa", "qb_vs_team"):
        out[f"{c}_diff"] = g[f"h_{c}"] - g[f"a_{c}"]
    return out
