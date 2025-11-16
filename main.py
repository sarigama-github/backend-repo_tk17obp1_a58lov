import os
from datetime import datetime, timedelta, timezone, date
from typing import List, Optional, Literal, Dict, Any

import pandas as pd
import numpy as np
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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


def _get_price_and_currency(symbol: str, as_of: Optional[date] = None) -> Dict[str, Any]:
    ticker = yf.Ticker(symbol)
    info = ticker.fast_info if hasattr(ticker, 'fast_info') else None
    ccy = None
    if info:
        ccy = getattr(info, 'currency', None) or info.get('currency') if isinstance(info, dict) else None
    # history
    if as_of is None:
        hist = ticker.history(period="1d", auto_adjust=False)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No price for {symbol}")
        price = float(hist["Close"].iloc[-1])
    else:
        # Get some buffer then pick last available on/before date
        hist = ticker.history(start=as_of - timedelta(days=5), end=as_of + timedelta(days=1), auto_adjust=False)
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No price for {symbol} on {as_of}")
        price = float(hist["Close"].ffill().iloc[-1])
    return {"price": price, "currency": ccy}


def _get_fx_rate(from_ccy: str, to_ccy: str, as_of: Optional[date] = None) -> float:
    if from_ccy.upper() == to_ccy.upper():
        return 1.0
    fx_ticker = _fx_symbol(from_ccy, to_ccy)
    data = yf.Ticker(fx_ticker).history(period="1d") if as_of is None else yf.Ticker(fx_ticker).history(start=as_of - timedelta(days=5), end=as_of + timedelta(days=1))
    if data.empty:
        # Try inverse
        inv = yf.Ticker(_fx_symbol(to_ccy, from_ccy)).history(period="1d")
        if inv.empty:
            raise HTTPException(status_code=404, detail=f"No FX for {from_ccy}->{to_ccy}")
        return 1.0 / float(inv["Close"].iloc[-1])
    return float(data["Close"].ffill().iloc[-1])


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
        # Average cost in trade currency weighted by quantity (buys positive, sells negative handled naturally)
        # Convert each leg to base currency using FX at trade date
        base_cost = 0.0
        total_qty = 0.0
        for _, row in grp.iterrows():
            leg_qty = float(row["quantity"])
            if abs(leg_qty) < 1e-12:
                continue
            px = float(row["price"])
            ccy = row["trade_currency"].upper()
            leg_date = row["trade_date"]
            fx = _get_fx_rate(ccy, BASE_CCY, as_of=leg_date)
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
        quote = _get_price_and_currency(p["symbol"], as_of=as_of)
        px = quote["price"]
        ccy = quote.get("currency") or BASE_CCY
        fx = _get_fx_rate(ccy, BASE_CCY, as_of=as_of)
        value_base = q * px * fx
        cost_base = q * p["avg_cost_base"]
        pnl_abs = value_base - cost_base
        pnl_pct = pnl_abs / cost_base if abs(cost_base) > 1e-12 else 0.0
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


@app.post("/api/holdings")
def add_holding(holding: HoldingIn):
    # Basic validation
    if not holding.symbol:
        raise HTTPException(400, "Symbol is required")
    # store
    create_document("holding", holding.model_dump())
    return {"status": "ok"}


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
