import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "train"))

from oof import run_oof

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "output" / "oof_result.parquet"


def run(start: str | None, end: str | None, model_names: list[str] | None, output: Path) -> None:
    t0 = time.time()
    result = run_oof(start=start, end=end, model_names=model_names)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output)
    print(f"done in {time.time() - t0:.0f}s, shape={result.shape}, saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, help="YYYY-MM-DD (default: full history)")
    parser.add_argument("--end", type=str, help="YYYY-MM-DD (default: today)")
    parser.add_argument(
        "--models", type=str, help="comma-separated FACTOR_MODELS keys (default: all registered)"
    )
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    run(
        args.start,
        args.end,
        args.models.split(",") if args.models else None,
        Path(args.output),
    )
