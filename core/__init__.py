"""
Core package for Nifty Options Algo

Contains main trading modules:
- data_handler: Market data fetching and processing
- signal_generation: Technical analysis and signal generation
- execution: Trade execution and position management
- strategy: Main strategy orchestration
- position: Data classes and enums
"""

import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nifty_options.log'),
        logging.StreamHandler()
    ]
)

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