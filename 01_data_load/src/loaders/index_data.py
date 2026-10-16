import pandas as pd

INDEX_TICKERS = {
    "^KS11": {"name": "KOSPI", "currency": "KRW"},
    "^GSPC": {"name": "S&P 500", "currency": "USD"},
}


def index_to_security(ticker: str) -> dict:
    info = INDEX_TICKERS[ticker]
    return {
        "ticker": ticker,
        "name": info["name"],
        "market": "INDEX",
        "sector": None,
        "listed_date": None,
        "currency": info["currency"],
    }


def index_price_df_to_rows(ticker: str, df: pd.DataFrame) -> list[dict]:
    df = df.dropna(subset=["Close"]).copy()
    if df.empty:
        return []
    df["daily_return"] = df["Close"].pct_change()

    rows = []
    for dt, r in df.iterrows():
        rows.append(
            {
                "ticker": ticker,
                "date": dt.date(),
                "close_price": float(r["Close"]),
                "adj_close": None,
                "market_cap": None,
                "volume": int(r["Volume"]) if pd.notna(r["Volume"]) else None,
                "daily_return": float(r["daily_return"]) if pd.notna(r["daily_return"]) else None,
            }
        )
    return rows
