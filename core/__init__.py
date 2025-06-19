"""
Core package for Nifty Options Algo

Contains main trading modules:
- data_handler: Market data fetching and processing
- signal_generation: Technical analysis and signal generation
- execution: Trade execution and position management
- strategy: Main strategy orchestration
- position: Data classes and enums
"""

from .data_handler import DataHandler
from .signal_generation import SignalGenerator
from .execution import ExecutionHandler
from .strategy import NiftyOptionsStrategy
from .position import Position, TradeSignal, OrderType, TransactionType, ProductType, ExchangeSegment

__all__ = [
    'DataHandler',
    'SignalGenerator',
    'ExecutionHandler',
    'NiftyOptionsStrategy',
    'Position',
    'TradeSignal',
    'OrderType',
    'TransactionType',
    'ProductType',
    'ExchangeSegment'
]