from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class OrderType(Enum):
    """Order types for Breeze Connect API"""
    MARKET = "market"
    LIMIT = "limit"
    STOPLOSS_LIMIT = "stoploss"
    STOPLOSS_MARKET = "stoploss_market"


class TransactionType(Enum):
    """Transaction types for Breeze Connect API"""
    BUY = "buy"
    SELL = "sell"


class ProductType(Enum):
    """Product types for Breeze Connect API"""
    INTRADAY = "intraday"
    DELIVERY = "delivery"
    MTF = "mtf"  # Margin Trading Financing
    CNC = "cnc"   # Cash and Carry
    CO = "co"     # Cover Order
    BO = "bo"     # Bracket Order


class ExchangeSegment(Enum):
    """Exchange segments for Breeze Connect API"""
    NSE = "NSE"
    NFO = "NFO"
    BSE = "BSE"
    BFO = "BFO"
    CDS = "CDS"
    MCX = "MCX"


@dataclass
class Position:
    """Position data class for better structure"""
    id: str
    order_id: Optional[str]
    type: str
    strike: int
    option_type: str
    quantity: int
    entry_price: float
    entry_timestamp: datetime
    status: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    return_pct: Optional[float] = None


@dataclass
class TradeSignal:
    """Trade signal data class"""
    signal_type: str
    entry_price: float
    stop_loss: float
    take_profit: Optional[float]
    strike: int
    option_type: str
    timestamp: datetime
    spot_price: float
    confidence: float = 1.0
    option_data: Optional[Dict[str, Any]] = None
