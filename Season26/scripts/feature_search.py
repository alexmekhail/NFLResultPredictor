"""
Which pre-game features predict NFL winners best?

    python Season26/scripts/feature_search.py             # team-stat groups, all combos
    python Season26/scripts/feature_search.py --extras    # + QB & injury groups (2016+)

Protocol (walk-forward, no peeking):
  - Predictions for season S use only data from S (games already played) and
    the 3 seasons before it: every feature is rebuilt from that slice, and the
    logistic regression trains on those 3 prior seasons.
  - Combinations are *ranked on validation seasons* only; the *holdout seasons*
    (default 2020-2025) are reported but never used to choose anything.
  - Accuracy = correct winner picks / games (ties excluded) at P = 0.5.

Writes models/reports/feature_search[_extras].md and a matching _combos.csv.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data import load_config
from src.feature_lab import (ALL_GROUPS, FEATURE_GROUPS, LOOKBACK_SEASONS, QB_INJURY_GROUPS,
                             all_combinations, group_columns, load_schedules,
                             load_team_stats, season_windows, walk_forward)


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
    lb = LOOKBACK_SEASONS
    val_start = args.val_start or (2016 if args.extras else 2010)
    val = list(range(val_start, args.val_end + 1))
    hold = list(range(args.val_end + 1, args.holdout_end + 1))
    tests = val + hold
    seasons = range(val_start - lb, args.holdout_end + 1)

    print(f"Loading {seasons.start}-{seasons.stop - 1} schedules + team stats…")
    sched = load_schedules(seasons, cache)
    stats = load_team_stats(seasons, cache)

    extras = None
    if args.extras:
        from src.qb_injury_features import extra_game_features
        extras = lambda sch, ss: extra_game_features(sch, cache, ss)
        print("Building QB + injury features (first run downloads ~250 MB)…")

    def windows(window=args.window, win_seasons=args.win_seasons):
        return season_windows(sched, stats, tests, lb, window, win_seasons, extras)

    wins = windows()
    groups = ALL_GROUPS if args.extras else FEATURE_GROUPS
    max_size = args.max_size or (4 if args.extras else None)

    # ── exhaustive search ────────────────────────────────────────────────────
    combos = list(all_combinations(groups, max_size))
    print(f"Scoring {len(combos)} feature combinations…")
    t0, rows = time.time(), []
    for i, combo in enumerate(combos, 1):
        cols = group_columns(combo)
        v = walk_forward(wins, cols, val)
        h = walk_forward(wins, cols, hold)
        rows.append({
            "features": " + ".join(combo), "n_groups": len(combo),
            "val_acc": v["accuracy"], "val_logloss": v["log_loss"],
            "holdout_acc": h["accuracy"], "holdout_logloss": h["log_loss"],
            "uses_spread": "spread" in combo,
            "uses_qb_injury": any(g in QB_INJURY_GROUPS for g in combo),
        })
        if i % 250 == 0:
            print(f"  {i}/{len(combos)}  ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(rows).sort_values(["val_acc", "val_logloss"], ascending=[False, True])
    reports = pathlib.Path(cfg["paths"]["reports"])
    suffix = "_extras" if args.extras else ""
    res.to_csv(reports / f"feature_search{suffix}_combos.csv", index=False)

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
    best_combo = tuple(res.iloc[0]["features"].split(" + "))
    no_spread = res[~res["uses_spread"]]
    best_ns = tuple(no_spread.iloc[0]["features"].split(" + "))

    hg = pd.concat([wins[s][wins[s]["season"] == s] for s in hold])
    always_home = hg["home_win"].mean()
    fav = hg[hg["spread_line"] != 0]
    pick_fav = ((fav["spread_line"] > 0).astype(int) == fav["home_win"]).mean()

    def disagreement(combo):
        r = walk_forward(wins, group_columns(combo), hold, return_preds=True)
        d = r["preds"].merge(hg[["game_id", "spread_line"]], on="game_id")
        d = d[d["spread_line"] != 0]
        model_home = d["prob_home"] >= 0.5
        dis = d[model_home != (d["spread_line"] > 0)]
        right = ((dis["prob_home"] >= 0.5).astype(int) == dis["home_win"]).mean() if len(dis) else float("nan")
        return len(dis) / len(d), right, len(dis)

    # ── what QB / injury features add to a few base models ──────────────────
    addon_md = ""
    if args.extras:
        bases = [("spread",), ("spread", "turnovers"), ("elo", "point_diff", "turnovers")]
        addons = [(g,) for g in QB_INJURY_GROUPS] + [
            ("qb_quality", "qb_vs_team", "qb_change"),
            ("injuries_snap", "injuries_count"),
            tuple(QB_INJURY_GROUPS),
        ]
        out = []
        for base in bases:
            b_v = walk_forward(wins, group_columns(base), val)["accuracy"]
            b_h = walk_forward(wins, group_columns(base), hold)["accuracy"]
            out.append({"model": " + ".join(base), "val acc": pct(b_v), "holdout acc": pct(b_h),
                        "Δ val": "", "Δ holdout": ""})
            for add in addons:
                cols = group_columns(base + add)
                v = walk_forward(wins, cols, val)["accuracy"]
                h = walk_forward(wins, cols, hold)["accuracy"]
                out.append({"model": "  + " + " + ".join(add), "val acc": pct(v), "holdout acc": pct(h),
                            "Δ val": f"{(v - b_v) * 100:+.1f}", "Δ holdout": f"{(h - b_h) * 100:+.1f}"})
        addon_md = md_table(pd.DataFrame(out))

    # ── sensitivity on the best combo ────────────────────────────────────────
    print("Sensitivity sweeps…")

    def sweep(label, values, run):
        out = []
        for v in values:
            w, tw = run(v)
            r_v = walk_forward(w, group_columns(best_combo), val, tw)
            r_h = walk_forward(w, group_columns(best_combo), hold, tw)
            out.append({label: v, "val acc": pct(r_v["accuracy"]), "holdout acc": pct(r_h["accuracy"]),
                        "holdout log loss": f"{r_h['log_loss']:.4f}"})
        return pd.DataFrame(out)

    tw_sweep = sweep("train window (seasons)", range(1, lb + 1), lambda v: (wins, v))
    win_combo = best_combo if "win_pct" in best_combo else best_combo + ("win_pct",)
    wp_rows = []
    for k in range(0, lb + 1):
        w = wins if k == args.win_seasons else windows(win_seasons=k)
        r_s = walk_forward(w, ["winpct_diff"], tests)
        r_v = walk_forward(w, group_columns(win_combo), val)
        r_h = walk_forward(w, group_columns(win_combo), hold)
        wp_rows.append({"win % lookback (prior seasons)": k,
                        "win % alone (all test seasons)": pct(r_s["accuracy"]),
                        f"{' + '.join(win_combo)} val": pct(r_v["accuracy"]),
                        f"{' + '.join(win_combo)} holdout": pct(r_h["accuracy"])})
    wp_sweep = pd.DataFrame(wp_rows)
    rw_sweep = sweep("rolling window (games)", [3, 5, 8, 12, 17],
                     lambda v: (wins if v == args.window else windows(window=v), lb))

    # Weights of the best model fit on the 3 seasons before the last holdout season.
    cols = ["home_field"] + group_columns(best_combo)
    last = hold[-1]
    tr = wins[last][wins[last]["season"] < last].dropna(subset=cols)
    lr = LogisticRegression(max_iter=1000).fit(StandardScaler().fit_transform(tr[cols]), tr["home_win"])
    weights = pd.DataFrame({"feature": cols, "weight (per 1 std dev)": np.round(lr.coef_[0], 3)}) \
        .sort_values("weight (per 1 std dev)", key=abs, ascending=False)

    # ── report ───────────────────────────────────────────────────────────────
    n_val = sum(int((wins[s]["season"] == s).sum()) for s in val)
    se = np.sqrt(0.65 * 0.35 / n_val)
    se_h = np.sqrt(0.65 * 0.35 / len(hg))
    dis_best, dis_best_right, n_dis_best = disagreement(best_combo)
    dis_ns, dis_ns_right, n_dis_ns = disagreement(best_ns)

    lines = [
        "# Feature search" + (" — with QB & injury features" if args.extras else ""),
        "",
        f"Generated by `scripts/feature_search.py{' --extras' if args.extras else ''}`. "
        f"**Lookback rule:** each season is predicted using only that season's earlier games "
        f"and the {lb} seasons before it — features (Elo, rolling stats, win %, QB and "
        f"injury history) are rebuilt from that slice, and the model trains on those {lb} seasons.",
        "",
        f"Validation seasons {val[0]}–{val[-1]} ({n_val} games) rank the combinations; holdout "
        f"seasons {hold[0]}–{hold[-1]} ({len(hg)} games) are only reported. Team stats are "
        f"rolling averages of the previous {args.window} games; win % covers the previous "
        f"{args.win_seasons} seasons plus the current season so far. {len(combos)} combinations "
        f"of {len(groups)} feature groups were scored"
        + (f" (up to {max_size} groups each)." if max_size else "."),
        "",
        f"Noise level: one standard error is about {pct(se)} on validation and {pct(se_h)} on "
        f"holdout, so gaps under ~{pct(2 * se_h)} on holdout are not meaningful.",
        "",
        "## Baselines (holdout)",
        "",
        f"- Always pick the home team: {pct(always_home)}",
        f"- Always pick the betting favorite: {pct(pick_fav)}",
        "",
    ]
    if addon_md:
        lines += ["## What QB and injury features add", "",
                  "Accuracy of three base models, then with each QB / injury group added. "
                  "Δ = change in percentage points.", "", addon_md, ""]
    lines += [
        "## Best combinations (ranked on validation)", "", md_table(fmt(res.head(15))), "",
        "## Best combinations without the betting line", "", md_table(fmt(no_spread.head(10))), "",
        "## Each feature group on its own", "", md_table(fmt(singles)), "",
        "## Disagreeing with the betting favorite (holdout)", "",
        f"- Best overall ({' + '.join(best_combo)}): picks against the favorite in "
        f"{pct(dis_best)} of games ({n_dis_best}); right in {pct(dis_best_right)} of those.",
        f"- Best without spread ({' + '.join(best_ns)}): picks against the favorite in "
        f"{pct(dis_ns)} of games ({n_dis_ns}); right in {pct(dis_ns_right)} of those.",
        "",
        f"## Sensitivity: training window ({' + '.join(best_combo)})", "", md_table(tw_sweep), "",
        "## Sensitivity: win % lookback", "", md_table(wp_sweep), "",
        f"## Sensitivity: rolling window for team stats ({' + '.join(best_combo)})", "",
        md_table(rw_sweep), "",
        f"## Best model's weights (trained on {last - lb}–{last - 1}, standardized)", "",
        "Positive = favors the home team when the home team is higher on that feature.", "",
        md_table(weights), "",
    ]
    out = reports / f"feature_search{suffix}.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extras", action="store_true", help="add QB & injury feature groups")
    ap.add_argument("--val-start", type=int, default=None,
                    help="first validation season (default 2010, or 2016 with --extras)")
    ap.add_argument("--val-end", type=int, default=2019)
    ap.add_argument("--holdout-end", type=int, default=2025)
    ap.add_argument("--window", type=int, default=8, help="rolling games for team stats")
    ap.add_argument("--win-seasons", type=int, default=3, help="prior seasons in win %%")
    ap.add_argument("--max-size", type=int, default=None,
                    help="max groups per combination (default all, or 4 with --extras)")
    main(ap.parse_args())
