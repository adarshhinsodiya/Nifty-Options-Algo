from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum


class OrderType(Enum):
    """Order types for better type safety"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class TransactionType(Enum):
    """Transaction types for better type safety"""
    BUY = "BUY"
    SELL = "SELL"


class ProductType(Enum):
    """Product types for better type safety"""
    INTRADAY = "INTRA"
    DELIVERY = "CNC"


class ExchangeSegment(Enum):
    """Exchange segments for better type safety"""
    NSE_EQ = "NSE_EQ"
    NSE_FNO = "NSE_FNO"
    INDEX = "IDX_I"


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
