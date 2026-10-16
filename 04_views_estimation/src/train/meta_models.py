from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from models import _scaled

# Registry pattern (see TRAIN.md): meta-model candidates, compared against each other by their
# own walk-forward OOF performance on the base models' OOF predictions (meta_oof.py). Kept
# deliberately simple -- the meta-level input is just a handful of correlated numbers (one per
# base model), where extra capacity (MLP, "modern tabular" models) is pure overfitting risk
# rather than learning opportunity, especially given the factor/sequence neural-net experiments
# already showed this kind of noisy financial target doesn't reward extra capacity anyway.
def _lgbm(num_leaves: int, n_estimators: int = 100) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=num_leaves,
        random_state=42,
        verbose=-1,
    )


META_MODELS = {
    "ridge": lambda: _scaled(Ridge(alpha=1.0, random_state=42)),
    # 3 num_leaves variants to empirically check the regularization sweet spot at this input
    # size (8 base-model predictions), same reasoning as the base-level RandomForest/LightGBM
    # tuning -- don't guess a single config, compare a few via run_meta_oof.py's existing loop.
    "lightgbm_leaves7": lambda: _lgbm(num_leaves=7),
    "lightgbm_leaves15": lambda: _lgbm(num_leaves=15),
    "lightgbm_leaves31": lambda: _lgbm(num_leaves=31),
}
