import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "train"))

from meta_data import build_meta_training_set
from meta_models import META_MODELS
from meta_oof import run_meta_oof 

DEFAULT_FACTOR = Path(__file__).resolve().parent.parent / "output" / "oof_result.parquet"
DEFAULT_SEQUENCE = Path(__file__).resolve().parent.parent / "output" / "sequence_oof_result.parquet"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def run(factor_path: str, sequence_path: str, output_dir: Path) -> None:
    print("merging factor + sequence OOF...", flush=True)
    meta_df = build_meta_training_set(factor_path, sequence_path)
    base_model_cols = [c for c in meta_df.columns if c not in ("as_of_date", "ticker", "target")]
    print(f"meta training set: {meta_df.shape}, base models: {base_model_cols}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_df.to_parquet(output_dir / "meta_training_set.parquet")

    for name in META_MODELS:
        result = run_meta_oof(meta_df, name)
        if result.empty:
            print(f"meta model [{name}]: no folds produced (not enough merged dates)", flush=True)
            continue
        corr = result["pred"].corr(result["target"])
        result.to_parquet(output_dir / f"meta_oof_{name}.parquet")
        print(f"meta model [{name}] OOF corr={corr:.4f} rows={len(result)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=str, default=str(DEFAULT_FACTOR))
    parser.add_argument("--sequence", type=str, default=str(DEFAULT_SEQUENCE))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    run(args.factor, args.sequence, Path(args.output_dir))
