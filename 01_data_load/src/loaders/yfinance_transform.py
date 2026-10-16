import pandas as pd


def universe_row_to_security(row: dict) -> dict:
    return {
        "ticker": row["ticker"],
        "name": row["name"],
        "market": "US",
        "sector": row.get("sector"),
        "listed_date": None,
        "currency": "USD",
    }


def price_df_to_rows(ticker: str, df: pd.DataFrame, shares_outstanding: float | None) -> list[dict]:
    df = df.dropna(subset=["Close"]).copy()
    if df.empty:
        return []
    df["daily_return"] = df["Adj Close"].pct_change()

    rows = []
    for dt, r in df.iterrows():
        adj_close = float(r["Adj Close"]) if pd.notna(r["Adj Close"]) else None
        rows.append(
            {
                "ticker": ticker,
                "date": dt.date(),
                "close_price": float(r["Close"]),
                "adj_close": adj_close,
                "market_cap": adj_close * shares_outstanding if adj_close is not None and shares_outstanding else None,
                "volume": int(r["Volume"]) if pd.notna(r["Volume"]) else None,
                "daily_return": float(r["daily_return"]) if pd.notna(r["daily_return"]) else None,
            }
        )
    return rows
