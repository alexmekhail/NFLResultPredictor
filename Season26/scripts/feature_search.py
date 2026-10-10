"""
Which pre-game features predict NFL winners best?

    python Season26/scripts/feature_search.py            # full search (~10 min)
    python Season26/scripts/feature_search.py --quick    # combos of up to 3 groups

Protocol (walk-forward, no peeking):
  - For each test season S, a logistic regression is trained on the
    `--train-window` seasons before S and scored on every game of S.
  - Combinations are *ranked on validation seasons* (default 2010-2019) only.
  - The *holdout seasons* (default 2020-2025) are reported but never used to
    choose anything, so the holdout number is an honest estimate.
  - Accuracy = correct winner picks / games (ties excluded) at P = 0.5.

Writes models/reports/feature_search.md and feature_search_combos.csv.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import time

import numpy as np
import pandas as pd

from src.data import load_config
from src.feature_lab import (FEATURE_GROUPS, all_combinations, game_matrix, group_columns,
                             load_schedules, load_team_stats, walk_forward)


def pct(x):
    return f"{x * 100:.1f}%"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main(args):
    cfg = load_config()
    cache = pathlib.Path(cfg["paths"]["interim"]) / "feature_lab"
    first_hist = args.val_start - 8          # history for rolling stats / win% / train window
    seasons = range(first_hist, args.holdout_end + 1)
    val = list(range(args.val_start, args.val_end + 1))
    hold = list(range(args.val_end + 1, args.holdout_end + 1))

    print(f"Loading {seasons.start}-{seasons.stop - 1} schedules + team stats…")
    sched = load_schedules(seasons, cache)
    sched = sched[sched["season"].between(seasons.start, seasons.stop - 1)]
    stats = load_team_stats(seasons, cache)
    games = game_matrix(sched, stats, window=args.window, win_seasons=args.win_seasons)

    # ── exhaustive search ────────────────────────────────────────────────────
    combos = list(all_combinations(max_size=3 if args.quick else None))
    print(f"Scoring {len(combos)} feature combinations…")
    t0, rows = time.time(), []
    for i, combo in enumerate(combos, 1):
        cols = group_columns(combo)
        v = walk_forward(games, cols, val, args.train_window)
        h = walk_forward(games, cols, hold, args.train_window)
        rows.append({
            "features": " + ".join(combo), "n_groups": len(combo),
            "val_acc": v["accuracy"], "val_logloss": v["log_loss"],
            "holdout_acc": h["accuracy"], "holdout_logloss": h["log_loss"],
            "uses_spread": "spread" in combo,
        })
        if i % 200 == 0:
            print(f"  {i}/{len(combos)}  ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(rows).sort_values(["val_acc", "val_logloss"], ascending=[False, True])
    reports = pathlib.Path(cfg["paths"]["reports"])
    reports.mkdir(parents=True, exist_ok=True)
    res.to_csv(reports / "feature_search_combos.csv", index=False)

    def fmt(df):
        out = df[["features", "val_acc", "holdout_acc", "val_logloss", "holdout_logloss"]].copy()
        for c in ("val_acc", "holdout_acc"):
            out[c] = out[c].map(pct)
        for c in ("val_logloss", "holdout_logloss"):
            out[c] = out[c].map(lambda x: f"{x:.4f}")
        return out.rename(columns={"val_acc": f"val acc {val[0]}-{val[-1]}",
                                   "holdout_acc": f"holdout acc {hold[0]}-{hold[-1]}",
                                   "val_logloss": "val log loss",
                                   "holdout_logloss": "holdout log loss"})

    singles = res[res["n_groups"] == 1].sort_values("val_acc", ascending=False)
    best = res.iloc[0]
    best_combo = tuple(best["features"].split(" + "))
    no_spread = res[~res["uses_spread"]]
    best_ns = tuple(no_spread.iloc[0]["features"].split(" + "))

    # Baselines on the holdout seasons.
    hg = games[games["season"].isin(hold)]
    always_home = hg["home_win"].mean()
    fav = hg[hg["spread_line"] != 0]
    pick_fav = ((fav["spread_line"] > 0).astype(int) == fav["home_win"]).mean()

    # How often each model disagrees with the betting favorite, and how it does then.
    def disagreement(combo):
        r = walk_forward(games, group_columns(combo), hold, args.train_window, return_preds=True)
        d = r["preds"].merge(games[["game_id", "spread_line"]], on="game_id")
        d = d[d["spread_line"] != 0]
        model_home = d["prob_home"] >= 0.5
        fav_home = d["spread_line"] > 0
        dis = d[model_home != fav_home]
        right = ((dis["prob_home"] >= 0.5).astype(int) == dis["home_win"]).mean() if len(dis) else float("nan")
        return len(dis) / len(d), right, len(dis)

    # ── sensitivity on the best combo ────────────────────────────────────────
    def sweep(label, values, build):
        out = []
        for v in values:
            g, tw = build(v)
            r_v = walk_forward(g, group_columns(best_combo), val, tw)
            r_h = walk_forward(g, group_columns(best_combo), hold, tw)
            out.append({label: v, f"val acc": pct(r_v["accuracy"]),
                        f"holdout acc": pct(r_h["accuracy"]),
                        "holdout log loss": f"{r_h['log_loss']:.4f}"})
        return pd.DataFrame(out)

    print("Sensitivity sweeps…")
    tw_sweep = sweep("train window (seasons)", [1, 2, 3, 4, 5, 8],
                     lambda v: (games, v))
    win_combo = best_combo if "win_pct" in best_combo else best_combo + ("win_pct",)
    wp_rows = []
    for k in [0, 1, 2, 3, 4, 5]:
        g = game_matrix(sched, stats, window=args.window, win_seasons=k)
        r_v = walk_forward(g, group_columns(win_combo), val, args.train_window)
        r_h = walk_forward(g, group_columns(win_combo), hold, args.train_window)
        r_s = walk_forward(g, ["winpct_diff"], val + hold, args.train_window)
        wp_rows.append({"win % lookback (prior seasons)": k,
                        "win % alone (all seasons)": pct(r_s["accuracy"]),
                        f"{' + '.join(win_combo)} val": pct(r_v["accuracy"]),
                        f"{' + '.join(win_combo)} holdout": pct(r_h["accuracy"])})
    wp_sweep = pd.DataFrame(wp_rows)
    rw_sweep = sweep("rolling window (games)", [3, 5, 8, 12, 17],
                     lambda v: (game_matrix(sched, stats, window=v, win_seasons=args.win_seasons),
                                args.train_window))

    # Fitted weights of the best model on the most recent window, standardized.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    cols = ["home_field"] + group_columns(best_combo)
    tr = games[games["season"].between(hold[-1] - args.train_window + 1, hold[-1])].dropna(subset=cols)
    lr = LogisticRegression(max_iter=1000).fit(StandardScaler().fit_transform(tr[cols]), tr["home_win"])
    weights = pd.DataFrame({"feature": cols, "weight (per 1 std dev)": np.round(lr.coef_[0], 3)}) \
        .sort_values("weight (per 1 std dev)", key=abs, ascending=False)

    # ── report ───────────────────────────────────────────────────────────────
    n_val = int(games["season"].isin(val).sum())
    se = np.sqrt(0.65 * 0.35 / n_val)
    dis_best, dis_best_right, n_dis_best = disagreement(best_combo)
    dis_ns, dis_ns_right, n_dis_ns = disagreement(best_ns)

    lines = [
        "# Feature search",
        "",
        f"Generated by `scripts/feature_search.py`. Validation seasons {val[0]}–{val[-1]} "
        f"({n_val} games) are used to rank combinations; holdout seasons {hold[0]}–{hold[-1]} "
        f"({len(hg)} games) are only reported. Each test season is predicted by a logistic "
        f"regression trained on the {args.train_window} seasons before it. Team stats are "
        f"rolling averages of the previous {args.window} games; win % covers the previous "
        f"{args.win_seasons} seasons plus the current season so far. "
        f"{len(combos)} combinations of {len(FEATURE_GROUPS)} feature groups were scored.",
        "",
        f"Noise level: one standard error on validation accuracy is about {pct(se)}, so "
        f"differences under ~{pct(2 * se)} between combinations are not meaningful.",
        "",
        "## Baselines (holdout)",
        "",
        f"- Always pick the home team: {pct(always_home)}",
        f"- Always pick the betting favorite: {pct(pick_fav)}",
        "",
        "## Best combinations (ranked on validation)",
        "",
        md_table(fmt(res.head(15))),
        "",
        "## Best combinations without the betting line",
        "",
        md_table(fmt(no_spread.head(10))),
        "",
        "## Each feature group on its own",
        "",
        md_table(fmt(singles)),
        "",
        "## Disagreeing with the betting favorite (holdout)",
        "",
        f"- Best overall ({' + '.join(best_combo)}): picks against the favorite in "
        f"{pct(dis_best)} of games ({n_dis_best}); right in {pct(dis_best_right)} of those.",
        f"- Best without spread ({' + '.join(best_ns)}): picks against the favorite in "
        f"{pct(dis_ns)} of games ({n_dis_ns}); right in {pct(dis_ns_right)} of those.",
        "",
        f"## Sensitivity: training window ({' + '.join(best_combo)})",
        "",
        md_table(tw_sweep),
        "",
        "## Sensitivity: win % lookback",
        "",
        md_table(wp_sweep),
        "",
        f"## Sensitivity: rolling window for team stats ({' + '.join(best_combo)})",
        "",
        md_table(rw_sweep),
        "",
        f"## Best model's weights (trained on {hold[-1] - args.train_window + 1}–{hold[-1]}, standardized)",
        "",
        "Positive = favors the home team when the home team is higher on that feature.",
        "",
        md_table(weights),
        "",
    ]
    out = reports / "feature_search.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-start", type=int, default=2010)
    ap.add_argument("--val-end", type=int, default=2019)
    ap.add_argument("--holdout-end", type=int, default=2025)
    ap.add_argument("--train-window", type=int, default=3)
    ap.add_argument("--window", type=int, default=8, help="rolling games for team stats")
    ap.add_argument("--win-seasons", type=int, default=3, help="prior seasons in win %%")
    ap.add_argument("--quick", action="store_true", help="only combos of up to 3 groups")
    main(ap.parse_args())
