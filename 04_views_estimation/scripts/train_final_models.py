import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "train"))

import joblib
import pandas as pd
import torch

from meta_data import build_meta_training_set
from meta_models import META_MODELS
from meta_oof import run_meta_oof
from model_registry import save_model_meta
from models import FACTOR_MODELS
from sequence_models import SEQUENCE_MODELS
from sequence_training_data import build_sequence_training_set
from training_data import build_training_set

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
OOF_FACTOR = Path(__file__).resolve().parent.parent / "output" / "oof_result.parquet"
OOF_SEQUENCE = Path(__file__).resolve().parent.parent / "output" / "sequence_oof_result.parquet"
NON_FEATURE_COLS = ("as_of_date", "ticker", "target")


def train_factor_models() -> None:
    print("building full factor training set...", flush=True)
    full = build_training_set()
    feature_cols = [c for c in full.columns if c not in NON_FEATURE_COLS]
    oof = pd.read_parquet(OOF_FACTOR)

    for name, factory in FACTOR_MODELS.items():
        t0 = time.time()
        model = factory()
        model.fit(full[feature_cols], full["target"])
        path = MODELS_DIR / f"factor_{name}.joblib"
        joblib.dump(model, path)

        model_oof = oof[oof["model"] == name]
        corr = model_oof["pred"].corr(model_oof["target"])
        save_model_meta(name, "factor", datetime.now(), float(corr), str(path))
        print(f"  [{name}] fit {time.time() - t0:.1f}s, oof_corr={corr:.4f}, saved to {path}", flush=True)


def train_sequence_models() -> None:
    print("building full sequence training set...", flush=True)
    _, _, X, y = build_sequence_training_set()
    oof = pd.read_parquet(OOF_SEQUENCE)

    for name, factory in SEQUENCE_MODELS.items():
        t0 = time.time()
        wrapper = factory()
        wrapper.fit(X, y)
        path = MODELS_DIR / f"sequence_{name}.pt"
        torch.save(wrapper.model.state_dict(), path)

        model_oof = oof[oof["model"] == name]
        corr = model_oof["pred"].corr(model_oof["target"])
        save_model_meta(name, "sequence", datetime.now(), float(corr), str(path))
        print(f"  [{name}] fit {time.time() - t0:.1f}s, oof_corr={corr:.4f}, saved to {path}", flush=True)


def train_meta_model() -> None:
    print("merging factor + sequence OOF for meta training...", flush=True)
    meta_df = build_meta_training_set(str(OOF_FACTOR), str(OOF_SEQUENCE))
    feature_cols = [c for c in meta_df.columns if c not in NON_FEATURE_COLS]

    best_name, best_corr, best_result = None, -float("inf"), None
    for name in META_MODELS:
        result = run_meta_oof(meta_df, name)
        if result.empty:
            print(f"  meta candidate [{name}]: no folds produced, skipping", flush=True)
            continue
        corr = result["pred"].corr(result["target"])
        print(f"  meta candidate [{name}] OOF corr={corr:.4f}", flush=True)
        if corr > best_corr:
            best_name, best_corr, best_result = name, corr, result

    if best_name is None:
        print("no meta model candidate produced any folds -- skipping meta training", flush=True)
        return

    print(f"winning meta model: {best_name} (OOF corr={best_corr:.4f})", flush=True)
    # Save the winner's OOF predictions -- same role as oof_result.parquet/sequence_oof_result.parquet
    # for the base models: a point-in-time-honest history of what the meta-model would have
    # predicted at each past date, usable for backtesting without needing to reload/retrain
    # anything (the final_model fit below, on ALL data, is for live/future inference only --
    # using IT to "predict" the past would be in-sample, not a real backtest).
    meta_oof_path = MODELS_DIR.parent / "output" / "meta_oof_result.parquet"
    best_result.to_parquet(meta_oof_path)
    print(f"saved meta OOF predictions (backtest artifact) to {meta_oof_path}", flush=True)
    final_model = META_MODELS[best_name]()
    final_model.fit(meta_df[feature_cols], meta_df["target"])
    path = MODELS_DIR / f"meta_{best_name}.joblib"
    joblib.dump(final_model, path)
    save_model_meta(best_name, "meta", datetime.now(), float(best_corr), str(path))
    print(f"saved final meta model to {path}", flush=True)


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== factor models ===", flush=True)
    train_factor_models()
    print("\n=== sequence models ===", flush=True)
    train_sequence_models()
    print("\n=== meta model ===", flush=True)
    train_meta_model()
    print("\ndone.", flush=True)
