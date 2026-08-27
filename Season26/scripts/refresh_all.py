# Ensure the season folder is importable (so `src` resolves) regardless of
# whether this is run from inside Season26/ or as `python Season26/scripts/refresh_all.py`.
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.train import main as train_main


def main():
    train_main()


if __name__ == "__main__":
    main()
