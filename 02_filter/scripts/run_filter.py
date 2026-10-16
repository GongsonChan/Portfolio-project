import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from universe import save_universe, select_universe

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", type=str, default=date.today().isoformat())
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--label", type=str, default="default", help="'default' = system's nightly universe; anything else = a custom one-off that won't overwrite it")
    args = parser.parse_args()

    selection = select_universe(args.as_of, n=args.n, label=args.label)
    save_universe(selection)
    print(f"{args.as_of} [{args.label}]: selected {len(selection)} tickers")
    print(selection.head(10).to_string(index=False))
