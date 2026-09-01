"""Build the static team assets the web dashboard uses:

  public/logos/<abbr>.png   one logo per team that appears in the schedule
  public/teams.json         { abbr: {name, nick, conf, division, color, color2} }

Run this once per season (team branding rarely changes). Logos are committed so
the site has no runtime dependency on an external CDN.

    python Season26/scripts/build_team_assets.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
import urllib.request

import nfl_data_py as nfl

from src.data import load_config, get_schedules

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PUBLIC = REPO_ROOT / "public"
LOGO_DIR = PUBLIC / "logos"


def main():
    cfg = load_config()
    season = int(cfg["season"])

    sched = get_schedules([season])
    abbrs = sorted(set(sched["home_team"]) | set(sched["away_team"]))

    teams = nfl.import_team_desc().set_index("team_abbr")

    LOGO_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for abbr in abbrs:
        if abbr not in teams.index:
            print(f"  ! {abbr}: not in team_desc, skipping")
            continue
        row = teams.loc[abbr]
        manifest[abbr] = {
            "name": row["team_name"],
            "nick": row["team_nick"],
            "conf": row["team_conf"],
            "division": row["team_division"],
            "color": row["team_color"],
            "color2": row["team_color2"],
        }
        # Pull a display-sized logo (≈160px) via ESPN's image combiner rather
        # than the full 500px asset, to keep the committed files small.
        src = row["team_logo_espn"].split("a.espncdn.com", 1)[-1]
        url = f"https://a.espncdn.com/combiner/i?img={src}&h=160&w=160"
        dest = LOGO_DIR / f"{abbr}.png"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
            f.write(r.read())
        print(f"  {abbr:4s} {row['team_name']:<24s} {dest.relative_to(REPO_ROOT)}  ({dest.stat().st_size // 1024} KB)")

    out = PUBLIC / "teams.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {out.relative_to(REPO_ROOT)} ({len(manifest)} teams)")


if __name__ == "__main__":
    main()
