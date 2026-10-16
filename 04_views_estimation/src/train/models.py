from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _imputed(model) -> Pipeline:
    """LightGBM handles NaN natively; RandomForest/Ridge/Lasso don't, so wrap them in a
    median-imputing pipeline (median, not mean, since several features like mom_* / vol_* are
    skewed by outlier tickers). Pipeline keeps the same fit/predict interface oof.py expects."""
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)])


def _scaled(model) -> Pipeline:
    """Ridge/Lasso/Huber additionally need feature scaling (tree models don't)."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


# Registry pattern (see TRAIN.md): every entry is a zero-arg factory returning a fresh,
# unfitted model with a sklearn-style fit(X, y)/predict(X) interface.
#
# Neural nets (MLP, MLP-deep, TabNet) were all tried and pulled back out. A 6-way MLP
# hyperparameter sweep (hidden sizes 16~128, alpha 1e-5~1e-1, early_stopping on/off) came back
# negative pred-target correlation across every single config -- not a tuning problem, just
# evidence that this feature set/target has no signal a neural net can extract that trees/linear
# models don't already get, consistent with tree-based models generally beating deep learning on
# tabular data (Grinsztajn et al. 2022). TabNet (CPU ~371s / GPU ~388s to fit 71K rows) was never
# even measured for accuracy once the MLP results came in -- not worth the time cost to confirm
# the same negative result from a related architecture.
#
# KNeighborsRegressor was also tried and dropped: 5 configs (k=10~500, uniform/distance) all came
# back negative correlation, getting worse as k grew -- same story as the neural nets, a different
# mechanism (local averaging) that just doesn't find signal here, likely compounded by the curse
# of dimensionality at 31 features. HuberRegressor (robust linear, different loss than Ridge/Lasso's
# squared error) was tried and kept: positive correlation, fast, and a genuinely different objective
# given monthly stock returns' fat tails.
FACTOR_MODELS = {
    # Unlike RandomForest below, this was already well-tuned: a num_leaves/n_estimators/
    # learning_rate sweep on the same fold 1 setup found every larger config *worse* than this
    # baseline (0.115 corr here vs 0.087-0.110 for bigger leaves/more trees). Makes sense --
    # LightGBM grows leaf-wise (picks the most useful split first) instead of RandomForest's
    # depth-wise growth, so it was already using its capacity efficiently at these settings.
    "lightgbm": lambda: LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    ),
    # max_depth=8/min_samples_leaf=50 (the original guess) underfit badly at this data scale:
    # a depth=None + min_samples_leaf=100 sweep on fold 1 (388K train rows) scored 0.121 OOF
    # corr vs 0.096 for the depth=8 baseline -- letting trees grow as deep as the leaf-size floor
    # allows works better than an arbitrary depth cap once there's enough data to support it.
    "random_forest": lambda: _imputed(
        RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=100,
            n_jobs=-1,
            random_state=42,
        )
    ),
    "ridge": lambda: _scaled(Ridge(alpha=1.0, random_state=42)),
    "lasso": lambda: _scaled(Lasso(alpha=0.001, random_state=42)),
    "huber": lambda: _scaled(HuberRegressor(epsilon=1.35, alpha=1e-4, max_iter=200)),
}
