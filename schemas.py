"""
Database Schemas for Portfolio App

Collections:
- holding: individual positions entered by the user
- pricecache: cached quotes by symbol & date to reduce external calls
- setting: user/app settings (e.g., base currency)
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import date, datetime


class Holding(BaseModel):
    """
    Collection: "holding"
    One document per transaction (buy/sell). Sells use negative quantity.
    currency: currency in which the trade was executed (e.g., EUR, USD)
    """
    symbol: str = Field(..., description="Ticker or identifier compatible with Yahoo Finance")
    name: Optional[str] = Field(None, description="Friendly name")
    isin: Optional[str] = Field(None, description="ISIN if available")
    asset_type: Optional[Literal["stock", "etf", "fund", "crypto", "other"]] = None

    quantity: float = Field(..., description="Number of units; negative for sell")
    price: float = Field(..., ge=0, description="Price per unit in trade currency")
    trade_currency: str = Field(..., min_length=3, max_length=3, description="ISO currency code, e.g., EUR, USD")

    trade_date: date = Field(..., description="Trade execution date")
    notes: Optional[str] = None


class PriceCache(BaseModel):
    """
    Collection: "pricecache"
    Cached adjusted close prices by symbol and date (UTC date). Currency as returned by Yahoo.
    """
    symbol: str
    yyyymmdd: str  # e.g., 2025-11-16
    currency: Optional[str] = None
    adj_close: float


class Setting(BaseModel):
    """
    Collection: "setting"
    Store global settings like base currency.
    """
    key: str
    value: str
