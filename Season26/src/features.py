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
    # Require everything except the label; unplayed games (spread known, no score
    # yet) survive with home_win = <NA>.
    return df[cols].dropna(subset=[c for c in cols if c != "home_win"])
