import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "jhgan/ko-sroberta-multitask"  # Korean-specific semantic-similarity model, decided
# in DESIGN.md's "열린 질문" -- local/free (no new API key), fits this project's scale (8-10
# holdings -- embedding runs in a couple seconds on CPU)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def compute_similarity_matrix(texts: dict[str, str]) -> pd.DataFrame:
    """texts: {ticker: "사업의 내용" text}. Returns a (ticker x ticker) cosine-similarity
    DataFrame -- business-model similarity, not return correlation (see DESIGN.md's "왜 필요한가"
    on how this complements 09_risk_management's component VaR)."""
    tickers = list(texts.keys())
    model = _get_model()
    embeddings = model.encode(list(texts.values()), normalize_embeddings=True)
    # Embeddings are L2-normalized, so the dot product IS the cosine similarity -- no need for a
    # separate cosine-distance step.
    sim = embeddings @ embeddings.T
    return pd.DataFrame(sim, index=tickers, columns=tickers)
