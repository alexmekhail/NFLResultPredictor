import pandas as pd

def basic_game_features(schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal baseline:
      - Home team indicator
      - Spread if available
      - Outcome label (NaN for games that haven't been played yet)
    """
    df = schedule.copy()

    # Label: 1 = home win, 0 = away win. Left as <NA> when the game has no final
    # score, so unplayed games can still be predicted but are never treated as
    # training labels (previously an unplayed game was silently labelled 0).
    if "home_score" in df.columns and "away_score" in df.columns:
        played = df["home_score"].notna() & df["away_score"].notna()
        df["home_win"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        df.loc[played, "home_win"] = (
            df.loc[played, "home_score"] > df.loc[played, "away_score"]
        ).astype(int)

    # Feature: spread (fallback = 0 if not present)
    if "spread_line" in df.columns:
        df["spread_home"] = -df["spread_line"]  # convert to home perspective
    else:
        df["spread_home"] = 0.0

    df["is_home"] = 1  # since we're predicting home perspective

    cols = ["game_id", "week", "home_team", "away_team", "home_win", "spread_home", "is_home"]
    if "spread_source" in df.columns:
        cols.append("spread_source")
    # Require everything except the label; unplayed games (spread known, no score
    # yet) survive with home_win = <NA>.
    return df[cols].dropna(subset=[c for c in cols if c not in ("home_win", "spread_source")])


def project_spreads(schedule: pd.DataFrame, before_week: int) -> pd.DataFrame:
    """
    Fill in `spread_line` for games whose line isn't published yet, so far-out
    weeks can be fed to the same spread-only model.

    Fits market power ratings from the lines already posted for weeks before
    `before_week`:
        spread_line ≈ hfa * (true home game) + rating[home] - rating[away]
    (nflverse spread_line is the home team's expected margin), solved by least
    squares with ratings constrained to sum to zero. Neutral-site games get no
    home-field term.

    Adds `spread_source`: "posted" for real lines, "projected" for filled ones.
    Rows that still can't be projected (team with no posted games) stay NaN.
    """
    import numpy as np

    df = schedule.copy()
    df["spread_source"] = np.where(df["spread_line"].notna(), "posted", None)

    hist = df[(df["week"] < before_week) & df["spread_line"].notna()]
    if hist.empty:
        return df

    teams = sorted(set(hist["home_team"]) | set(hist["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    def is_home_game(frame):
        if "location" in frame.columns:
            return (frame["location"] != "Neutral").astype(float).values
        return np.ones(len(frame))

    # Design: one column per team rating + a home-field column.
    A = np.zeros((len(hist) + 1, n + 1))
    for r, (h, a) in enumerate(zip(hist["home_team"], hist["away_team"])):
        A[r, idx[h]] = 1.0
        A[r, idx[a]] = -1.0
    A[:len(hist), n] = is_home_game(hist)
    A[len(hist), :n] = 1.0  # sum-to-zero constraint row
    b = np.append(hist["spread_line"].values, 0.0)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    rating, hfa = coef[:n], coef[n]

    todo = df["spread_line"].isna() & df["home_team"].isin(idx) & df["away_team"].isin(idx)
    if todo.any():
        sub = df[todo]
        proj = (
            hfa * is_home_game(sub)
            + sub["home_team"].map(lambda t: rating[idx[t]]).values
            - sub["away_team"].map(lambda t: rating[idx[t]]).values
        )
        # Round to the half point, like a real line.
        df.loc[todo, "spread_line"] = np.round(proj * 2) / 2
        df.loc[todo, "spread_source"] = "projected"
    return df
