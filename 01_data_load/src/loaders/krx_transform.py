from .krx_client import to_date, to_num


def base_info_to_security(row: dict, market: str) -> dict:
    return {
        "ticker": row["ISU_SRT_CD"],
        "name": row["ISU_NM"],
        "market": market,
        "sector": row["SECT_TP_NM"] or None,
        "listed_date": to_date(row["LIST_DD"]),
        "currency": "KRW",
    }


def trade_row_to_security_stub(row: dict, market: str) -> dict:
    # NOTE: unlike base_info, ISU_CD here is actually the 6-digit short code (not the ISIN)
    return {
        "ticker": row["ISU_CD"],
        "name": row["ISU_NM"],
        "market": market,
        "currency": "KRW",
    }


def trade_row_to_price(row: dict) -> dict:
    fluc_rt = to_num(row.get("FLUC_RT"))
    return {
        "ticker": row["ISU_CD"],
        "date": to_date(row["BAS_DD"]),
        "close_price": to_num(row.get("TDD_CLSPRC")),
        "adj_close": None,
        "market_cap": to_num(row.get("MKTCAP")),
        "volume": to_num(row.get("ACC_TRDVOL")),
        "daily_return": fluc_rt / 100 if fluc_rt is not None else None,
    }
