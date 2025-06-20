import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
import configparser

# Import position dataclass
from core.position import TradeSignal


class SignalGenerator:
    """
    Analyzes market data and generates trading signals based on candle patterns
    and technical indicators
    """
    
    def __init__(self, config: configparser.ConfigParser, logger: Optional[logging.Logger] = None):
        """
        Initialize the signal generator
        
        Args:
            config: Configuration parser
            logger: Logger instance (optional)
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Get strategy configuration
        self.strategy_name = config.get('strategy', 'strategy_name', fallback='candle_pattern')
        self.min_volume = int(config.get('strategy', 'min_volume_filter', fallback='1000'))
        self.volatility_threshold = float(config.get('strategy', 'volatility_threshold', fallback='0.05'))
        
        # Initialize technical indicators
        self.indicators = {
            'rsi': False,
            'macd': False,
            'bollinger': False
        }
        
        # Load strategy parameters from config
        self.confidence_threshold = float(config.get('strategy', 'confidence_threshold', fallback='0.7'))
        self.stop_loss_pct = float(config.get('strategy', 'stop_loss_pct', fallback='0.3'))
        self.take_profit_pct = float(config.get('strategy', 'take_profit_pct', fallback='0.6'))
        self.strike_step = float(config.get('strategy', 'strike_step', fallback='50.0'))
        self.ist_tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        
        # Initialize signal history
        self.signal_history = []
        self.max_history_size = int(config.get('strategy', 'max_signal_history', fallback='100'))
    
    def add_signal_to_history(self, signal: TradeSignal) -> None:
        """
        Add a signal to the history and maintain the maximum size
        
        Args:
            signal: Trade signal to add
        """
        self.signal_history.append(signal)
        
        # Trim history if needed
        if len(self.signal_history) > self.max_history_size:
            self.signal_history = self.signal_history[-self.max_history_size:]
    
    def analyze_candle_pattern(self, df: pd.DataFrame, index: int) -> Tuple[Optional[str], Optional[TradeSignal]]:
        """
        Analyze 2-candle pattern (previous + current) for trading signals.
        """
        if index < 0:
            index = len(df) + index
        if index >= len(df) or index < 1:
            self.logger.debug("Not enough candles for analysis")
            return None, None

        try:
            current = df.iloc[index]
            prev = df.iloc[index - 1]
        except IndexError:
            self.logger.error("Invalid index for current or previous candle")
            return None, None

        for candle in [current, prev]:
            if any(pd.isna(x) for x in [candle['open'], candle['close'], candle['high'], candle['low']]):
                self.logger.error("Missing candle data")
                return None, None

        prev_body = abs(prev['close'] - prev['open'])
        prev_range = prev['high'] - prev['low']
        if prev_range == 0:
            self.logger.error("Prev candle range is zero")
            return None, None

        if prev['close'] < prev['open']:
            prev_top_wick = prev['high'] - prev['open']
            prev_bottom_wick = prev['close'] - prev['low']
        else:
            prev_top_wick = prev['high'] - prev['close']
            prev_bottom_wick = prev['open'] - prev['low']
            
        signal_type = None
        entry_price = current['open']
        
        if (
            prev['close'] < prev['open'] and
            (prev_top_wick > prev_body) and
            (prev_top_wick > prev_bottom_wick) and
            (current['low'] < prev['low']) and
            (current['close'] > prev['open']) and
            (current['close'] > current['open'])
        ):
            signal_type = "LONG"

        # SHORT signal conditions
        elif (
            prev['close'] > prev['open'] and
            (prev_bottom_wick > prev_body) and
            (prev_bottom_wick > prev_top_wick) and
            (current['high'] > prev['high']) and
            (current['close'] < prev['open']) and
            (current['close'] < current['open'])
        ):
            signal_type = "SHORT"

        if signal_type:
            risk = abs(entry_price - prev['low']) if signal_type == "LONG" else abs(entry_price - prev['high'])
            profit_target = risk * 1.5
            take_profit = entry_price + profit_target if signal_type == "LONG" else entry_price - profit_target
            strike = int(round(prev['close'] / self.strike_step) * self.strike_step)
            option_type = "ce" if signal_type == "LONG" else "pe"
            timestamp = datetime.now(self.ist_tz)

            return signal_type, TradeSignal(
                signal_type=signal_type,
                entry_price=entry_price,
                stop_loss=prev['low'] if signal_type == "LONG" else prev['high'],
                take_profit=take_profit,
                strike=strike,
                option_type=option_type,
                timestamp=timestamp,
                spot_price=entry_price
            )

        return None, None
    
    def generate_signals(self, df: pd.DataFrame, spot_price: float) -> List[TradeSignal]:
        """
        Generate trading signals based on price data analysis
        
        Args:
            df: DataFrame with OHLCV price data
            spot_price: Current NIFTY spot price
            
        Returns:
            List of trade signals
        """
        signals = []
        
        # Analyze latest candle patterns
        candle_signal_type, candle_signal = self.analyze_candle_pattern(df, -1)
        if candle_signal:
            signals.append(candle_signal)
            
        # Filter signals based on strategy rules
        filtered_signals = self.filter_signals(signals)
        
        # Log generated signals
        if filtered_signals:
            self.logger.info(f"Generated {len(filtered_signals)} signals")
            
        return filtered_signals
    
    def filter_signals(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """
        Filter signals based on various criteria
        
        Args:
            signals: List of generated signals
            
        Returns:
            Filtered list of signals
        """
        if not signals:
            return []
            
        filtered_signals = []
        
        for signal in signals:
            # Skip signals that don't meet minimum requirements
            if not all([signal.signal_type, signal.entry_price > 0, signal.strike > 0]):
                continue
                
            # Check for recent similar signals
            is_duplicate = False
            for hist_signal in self.signal_history[-5:]:  # Check last 5 signals
                if (
                    signal.signal_type == hist_signal.signal_type and
                    (signal.timestamp - hist_signal.timestamp).total_seconds() < 3600  # 1 hour
                ):
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                filtered_signals.append(signal)
                self.add_signal_to_history(signal)
        
        return filtered_signals