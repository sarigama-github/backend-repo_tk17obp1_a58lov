import os
from datetime import datetime, timedelta, timezone, date
from typing import List, Optional, Literal, Dict, Any

import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import re

from database import db, create_document, get_documents

app = FastAPI(title="Portfolio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Models (requests) ----------
class HoldingIn(BaseModel):
    symbol: str
    name: Optional[str] = None
    isin: Optional[str] = None
    asset_type: Optional[Literal["stock", "etf", "fund", "crypto", "other"]] = None
    quantity: float
    price: float
    trade_currency: str
    trade_date: date
    notes: Optional[str] = None


class RangeQuery(BaseModel):
    period: Literal["today", "yesterday", "week", "month", "year", "ytd", "1y", "all"] = "ytd"


BASE_CCY = os.getenv("BASE_CURRENCY", "EUR").upper()


# ---------- Utilities ----------

_YF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}


TICKER_PATTERN = re.compile(r"^[A-Z0-9\.=\-]{1,15}$")


def _utc_date(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).date().isoformat()


def _period_to_dates(period: str):
    now = datetime.now(timezone.utc)
    today = now.date()
    if period == "today":
        start = today
    elif period == "yesterday":
        start = today - timedelta(days=1)
        today = start
    elif period == "week":
        start = today - timedelta(days=7)
    elif period == "month":
        start = today - timedelta(days=30)
    elif period in ("year", "1y"):
        start = today - timedelta(days=365)
    elif period == "ytd":
        start = date(today.year, 1, 1)
    else:  # all
        start = date(1970, 1, 1)
    return start, today


def _fx_symbol(from_ccy: str, to_ccy: str) -> str:
    if from_ccy.upper() == to_ccy.upper():
        return ""
    # Yahoo finance FX tickers like EURUSD=X
    return f"{from_ccy.upper()}{to_ccy.upper()}=X"


def _yahoo_chart_price(symbol: str, as_of: Optional[date] = None) -> Dict[str, Any]:
    """Fetch price and currency from Yahoo Chart API as a fallback to yfinance.
    Returns dict with keys price (float) and currency (str|None).
    """
    base = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol
    params: Dict[str, Any]
    if as_of is None:
        params = {"range": "5d", "interval": "1d"}
    else:
        # Use one-week window around the date to ensure we get a candle, then pick last <= as_of
        start_dt = datetime.combine(as_of - timedelta(days=5), datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt = datetime.combine(as_of + timedelta(days=1), datetime.min.time()).replace(tzinfo=timezone.utc)
        params = {
            "period1": int(start_dt.timestamp()),
            "period2": int(end_dt.timestamp()),
            "interval": "1d",
        }
    try:
        r = requests.get(base, params=params, headers=_YF_HEADERS, timeout=12)
        r.raise_for_status()
        j = r.json()
        result = (j or {}).get("chart", {}).get("result") or []
        if not result:
            raise ValueError("No chart result")
        meta = result[0].get("meta", {})
        currency = meta.get("currency")
        ts = result[0].get("timestamp") or []
        close = result[0].get("indicators", {}).get("quote", [{}])[0].get("close") or []
        if not ts or not close:
            raise ValueError("No timestamp/close")
        # Build list of (time, close) and pick last available <= as_of (or last overall)
        data = [(datetime.fromtimestamp(t, tz=timezone.utc).date(), c) for t, c in zip(ts, close) if c is not None]
        if not data:
            raise ValueError("No close data")
        if as_of is None:
            price = float(data[-1][1])
        else:
            # filter <= as_of
            cand = [c for d, c in data if d <= as_of]
            if not cand:
                raise ValueError("No data on/before date")
            price = float(cand[-1])
        return {"price": price, "currency": currency}
    except Exception:
        raise HTTPException(status_code=404, detail=f"No price for {symbol}{' on ' + as_of.isoformat() if as_of else ''}")


def _get_price_and_currency(symbol: str, as_of: Optional[date] = None) -> Dict[str, Any]:
    # First attempt: yfinance (fast when available)
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info if hasattr(ticker, 'fast_info') else None
        ccy = None
        if info:
            # fast_info may be object-like or dict-like across versions
            ccy = getattr(info, 'currency', None)
            if ccy is None and isinstance(info, dict):
                ccy = info.get('currency')
        if as_of is None:
            hist = ticker.history(period="1d", auto_adjust=False)
            if hist is not None and not hist.empty:
                price = float(hist["Close"].iloc[-1])
                return {"price": price, "currency": ccy}
        else:
            hist = ticker.history(start=as_of - timedelta(days=5), end=as_of + timedelta(days=1), auto_adjust=False)
            if hist is not None and not hist.empty:
                price = float(hist["Close"].ffill().iloc[-1])
                return {"price": price, "currency": ccy}
    except Exception:
        pass
    # Fallback: direct Yahoo chart API
    return _yahoo_chart_price(symbol, as_of=as_of)


def _get_fx_rate(from_ccy: str, to_ccy: str, as_of: Optional[date] = None) -> float:
    if from_ccy.upper() == to_ccy.upper():
        return 1.0
    fx_ticker = _fx_symbol(from_ccy, to_ccy)
    # Try yfinance first
    try:
        data = yf.Ticker(fx_ticker).history(period="1d") if as_of is None else yf.Ticker(fx_ticker).history(start=as_of - timedelta(days=5), end=as_of + timedelta(days=1))
        if data is not None and not data.empty:
            return float(data["Close"].ffill().iloc[-1])
        # Try inverse
        inv = yf.Ticker(_fx_symbol(to_ccy, from_ccy)).history(period="1d")
        if inv is not None and not inv.empty:
            return 1.0 / float(inv["Close"].ffill().iloc[-1])
    except Exception:
        pass
    # Fallback to Yahoo chart API for FX
    try:
        fx_meta = _yahoo_chart_price(fx_ticker, as_of=as_of)
        return float(fx_meta["price"])
    except HTTPException:
        # Try inverse via chart API
        inv_meta = _yahoo_chart_price(_fx_symbol(to_ccy, from_ccy), as_of=as_of)
        return 1.0 / float(inv_meta["price"])


def _yahoo_search(query: str, quotes: int = 10) -> List[Dict[str, Any]]:
    """Robust Yahoo search with fallback to autocomplete API.
    Returns up to `quotes` results with symbol, name, exchange, type, currency.
    """
    q = (query or "").strip()
    if not q:
        return []

    headers = _YF_HEADERS

    # Primary endpoint
    url1 = "https://query1.finance.yahoo.com/v1/finance/search"
    params1 = {
        "q": q,
        "quotesCount": quotes,
        "newsCount": 0,
        "listsCount": 0,
        "enableFuzzyQuery": "true",
        "lang": "es-ES",
        "region": "ES",
    }

    def _norm_item(sym: Optional[str], name: Optional[str], exch: Optional[str], qtype: Optional[str], ccy: Optional[str]):
        return {
            "symbol": sym,
            "shortname": name,
            "exchange": exch,
            "type": qtype,
            "currency": ccy,
        }

    results: List[Dict[str, Any]] = []
    try:
        r = requests.get(url1, params=params1, timeout=8, headers=headers)
        r.raise_for_status()
        data = r.json() or {}
        for it in (data.get("quotes") or [])[:quotes]:
            results.append(
                _norm_item(
                    it.get("symbol"),
                    it.get("shortname") or it.get("longname") or it.get("name"),
                    it.get("exchDisp") or it.get("exchange"),
                    it.get("quoteType"),
                    it.get("currency"),
                )
            )
    except Exception:
        results = []

    # Fallback if empty: query2 autocomplete API
    if not results:
        url2 = "https://query2.finance.yahoo.com/v6/finance/autocomplete"
        params2 = {"query": q, "lang": "es-ES", "region": "ES"}
        try:
            r2 = requests.get(url2, params=params2, timeout=8, headers=headers)
            r2.raise_for_status()
            data2 = r2.json() or {}
            for it in (data2.get("ResultSet", {}).get("Result") or [])[:quotes]:
                results.append(
                    _norm_item(
                        it.get("symbol"),
                        it.get("name"),
                        it.get("exchDisp") or it.get("exch") or it.get("exchDisp"),
                        it.get("typeDisp") or it.get("type"),
                        it.get("exch"),  # currency not provided here; keep None or exch code
                    )
                )
        except Exception:
            results = []

    # Heuristic: if user typed something like "NXT" and we're in Spain, also try .MC suffix
    if not results and q.isalpha() and len(q) <= 5:
        for suffix in [".MC", ".MA", ".LS", ".PA", ".MI"]:
            sym_try = (q.upper() + suffix)
            results.append(_norm_item(sym_try, None, None, None, None))
            if len(results) >= quotes:
                break

    # Deduplicate by symbol, keep order
    seen = set()
    deduped = []
    for it in results:
        sym = it.get("symbol")
        if sym and sym not in seen:
            seen.add(sym)
            deduped.append(it)
    return deduped[:quotes]


def _compute_positions() -> List[Dict[str, Any]]:
    txs = get_documents("holding", {})
    df = pd.DataFrame(txs)
    if df.empty:
        return []
    df["symbol"] = df["symbol"].astype(str)
    df["quantity"] = df["quantity"].astype(float)
    df["price"] = df["price"].astype(float)
    df["trade_currency"] = df["trade_currency"].astype(str)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

    positions = []
    for sym, grp in df.groupby("symbol"):
        qty = grp["quantity"].sum()
        if abs(qty) < 1e-9:
            continue
        # Weighted cost in base currency using FX at trade date
        base_cost = 0.0
        total_qty = 0.0
        for _, row in grp.iterrows():
            leg_qty = float(row["quantity"])
            if abs(leg_qty) < 1e-12:
                continue
            px = float(row["price"])
            ccy = row["trade_currency"].upper()
            leg_date = row["trade_date"]
            try:
                fx = _get_fx_rate(ccy, BASE_CCY, as_of=leg_date)
            except Exception:
                fx = 1.0  # fallback to 1 if FX unavailable
            base_cost += leg_qty * px * fx
            total_qty += leg_qty
        avg_cost_base = base_cost / total_qty if abs(total_qty) > 1e-12 else 0.0
        positions.append({
            "symbol": sym,
            "name": grp["name"].dropna().iloc[0] if grp["name"].notna().any() else sym,
            "quantity": float(qty),
            "avg_cost_base": float(avg_cost_base),
        })
    return positions


def _portfolio_value(as_of: Optional[date] = None) -> Dict[str, Any]:
    positions = _compute_positions()
    total_value = 0.0
    items = []
    for p in positions:
        q = p["quantity"]
        try:
            quote = _get_price_and_currency(p["symbol"], as_of=as_of)
            px = float(quote.get("price", 0.0))
            ccy = (quote.get("currency") or BASE_CCY)
            fx = _get_fx_rate(ccy, BASE_CCY, as_of=as_of)
        except Exception:
            px = 0.0
            ccy = BASE_CCY
            fx = 1.0
        value_base = q * px * fx
        cost_base = q * p["avg_cost_base"]
        pnl_abs = value_base - cost_base
        pnl_pct = (pnl_abs / cost_base) if abs(cost_base) > 1e-12 else 0.0
        items.append({
            "symbol": p["symbol"],
            "name": p.get("name", p["symbol"]),
            "quantity": q,
            "price": px,
            "price_ccy": ccy,
            "fx": fx,
            "value_base": value_base,
            "cost_base": cost_base,
            "pnl_abs": pnl_abs,
            "pnl_pct": pnl_pct,
        })
        total_value += value_base
    return {"base_currency": BASE_CCY, "total_value": total_value, "positions": items}


# ---------- Routes ----------
@app.get("/")
def root():
    return {"message": "Portfolio API running", "base_currency": BASE_CCY}


@app.get("/api/validate-ticker")
def validate_ticker(symbol: str = Query(..., min_length=1)):
    """Validate a Yahoo Finance symbol and return metadata.
    Prefer live price; if unavailable, accept plausible symbols so the user can continue.
    """
    sym = symbol.strip().upper()
    # First, try to fetch a real price
    try:
        quote = _get_price_and_currency(sym)
        meta = {"symbol": sym, "currency": quote.get("currency"), "price": quote.get("price")}
        suggestions = _yahoo_search(sym, quotes=1)
        if suggestions:
            meta["shortname"] = suggestions[0].get("shortname")
            meta["exchange"] = suggestions[0].get("exchange")
        return {"valid": True, **meta}
    except Exception:
        # No live data access. Fall through to permissive checks below.
        pass

    # If search confirms the exact symbol, accept it without price
    try:
        suggestions = _yahoo_search(sym, quotes=10)
        if any((s.get("symbol") or "").upper() == sym for s in suggestions):
            return {"valid": True, "symbol": sym}
    except Exception:
        pass

    # As last resort, accept plausible ticker patterns so the user can save manually
    if TICKER_PATTERN.match(sym):
        return {"valid": True, "symbol": sym}

    raise HTTPException(404, detail=f"Ticker '{sym}' no encontrado o sin datos")


@app.get("/api/suggest")
def suggest(q: str = Query(..., min_length=1), limit: int = 8):
    """Autocomplete suggestions from Yahoo search."""
    results = _yahoo_search(q.strip(), quotes=limit)
    return {"query": q, "results": results}


@app.post("/api/holdings")
def add_holding(holding: HoldingIn):
    # Basic validation and normalization
    if not holding.symbol:
        raise HTTPException(400, "Symbol is required")
    symbol_norm = holding.symbol.strip().upper()

    # Soft-validate: ensure ticker format is plausible; skip live price validation to avoid blocking
    if not TICKER_PATTERN.match(symbol_norm):
        raise HTTPException(400, detail="El formato del ticker no es válido")

    # Ensure Mongo-serializable payload (convert date -> datetime)
    payload = holding.model_dump()
    payload["symbol"] = symbol_norm
    payload["trade_currency"] = (payload.get("trade_currency") or "EUR").upper()
    if isinstance(payload.get("trade_date"), date):
        # store as UTC datetime at 00:00
        payload["trade_date"] = datetime.combine(payload["trade_date"], datetime.min.time()).replace(tzinfo=timezone.utc)
    # store
    create_document("holding", payload)
    return {"status": "ok", "symbol": symbol_norm}


@app.get("/api/portfolio")
def get_portfolio(as_of: Optional[date] = Query(default=None)):
    return _portfolio_value(as_of=as_of)


@app.get("/api/summary")
def get_summary(period: Literal["today", "yesterday", "week", "month", "year", "ytd", "1y", "all"] = "ytd"):
    start, end = _period_to_dates(period)
    # Value now
    now_val = _portfolio_value(as_of=end)
    # Value at start
    start_val = _portfolio_value(as_of=start)
    delta = now_val["total_value"] - start_val["total_value"]
    pct = delta / start_val["total_value"] if start_val["total_value"] else 0.0
    return {
        "base_currency": BASE_CCY,
        "period": period,
        "from": start.isoformat(),
        "to": end.isoformat(),
        "value_now": now_val["total_value"],
        "value_then": start_val["total_value"],
        "pnl_abs": delta,
        "pnl_pct": pct,
    }


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            collections = db.list_collection_names()
            response["collections"] = collections[:10]
            response["database"] = "✅ Connected & Working"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
