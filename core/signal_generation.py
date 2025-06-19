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
    
    def analyze_technical_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze technical indicators in the price data
        
        Args:
            df: DataFrame with OHLCV price data
            
        Returns:
            Dictionary of detected signals and their confidence scores
        """
        signals = {}
        
        # Ensure we have enough data
        if len(df) < 20:
            self.logger.warning("Not enough data for technical analysis")
            return signals
        
        # Calculate moving averages
        df['SMA5'] = df['Close'].rolling(window=5).mean()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Get the last values
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else None
        
        # Detect moving average crossover (bullish)
        if prev_row is not None and prev_row['SMA5'] <= prev_row['SMA20'] and last_row['SMA5'] > last_row['SMA20']:
            # Calculate confidence based on the strength of the crossover
            crossover_strength = (last_row['SMA5'] - last_row['SMA20']) / last_row['SMA20']
            confidence = min(1.0, crossover_strength * 100)
            signals['ma_crossover_bullish'] = confidence
        
        # Detect moving average crossover (bearish)
        if prev_row is not None and prev_row['SMA5'] >= prev_row['SMA20'] and last_row['SMA5'] < last_row['SMA20']:
            # Calculate confidence based on the strength of the crossover
            crossover_strength = (last_row['SMA20'] - last_row['SMA5']) / last_row['SMA20']
            confidence = min(1.0, crossover_strength * 100)
            signals['ma_crossover_bearish'] = confidence
        
        # Detect RSI overbought/oversold
        if last_row['RSI'] > 70:
            confidence = min(1.0, (last_row['RSI'] - 70) / 30)
            signals['rsi_overbought'] = confidence
        elif last_row['RSI'] < 30:
            confidence = min(1.0, (30 - last_row['RSI']) / 30)
            signals['rsi_oversold'] = confidence
        
        return signals
    
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
        
        # Analyze candle patterns
        patterns = {}
        
        # Analyze technical indicators
        indicators = self.analyze_technical_indicators(df)
        
        # Combine patterns and indicators
        all_signals = {**patterns, **indicators}
        
        # Log detected patterns and indicators
        if patterns:
            self.logger.info(f"Detected patterns: {patterns}")
        if indicators:
            self.logger.info(f"Detected indicators: {indicators}")
        
        # Generate signals based on the strategy
        if self.strategy_name == 'candle_pattern':
            # Bullish signals
            bullish_patterns = ['bullish_engulfing', 'hammer', 'morning_star', 'ma_crossover_bullish', 'rsi_oversold']
            bullish_confidence = 0.0
            for pattern in bullish_patterns:
                if pattern in all_signals and all_signals[pattern] > bullish_confidence:
                    bullish_confidence = all_signals[pattern]
            
            # Bearish signals
            bearish_patterns = ['bearish_engulfing', 'shooting_star', 'evening_star', 'ma_crossover_bearish', 'rsi_overbought']
            bearish_confidence = 0.0
            for pattern in bearish_patterns:
                if pattern in all_signals and all_signals[pattern] > bearish_confidence:
                    bearish_confidence = all_signals[pattern]
            
            # Generate signals if confidence is above threshold
            timestamp = datetime.now()
            
            if bullish_confidence > self.confidence_threshold:
                # Bullish signal - buy put options
                entry_price = spot_price
                stop_loss = entry_price * (1 + self.stop_loss_pct)
                take_profit = entry_price * (1 - self.take_profit_pct)
                
                signal = TradeSignal(
                    signal_type="BUY_PUT",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strike=0,  # Will be determined by execution module
                    option_type="PE",
                    timestamp=timestamp,
                    spot_price=spot_price,
                    confidence=bullish_confidence
                )
                
                signals.append(signal)
                self.add_signal_to_history(signal)
                self.logger.info(f"Generated bullish signal: {signal}")
            
            if bearish_confidence > self.confidence_threshold:
                # Bearish signal - buy call options
                entry_price = spot_price
                stop_loss = entry_price * (1 - self.stop_loss_pct)
                take_profit = entry_price * (1 + self.take_profit_pct)
                
                signal = TradeSignal(
                    signal_type="BUY_CALL",
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    strike=0,  # Will be determined by execution module
                    option_type="CE",
                    timestamp=timestamp,
                    spot_price=spot_price,
                    confidence=bearish_confidence
                )
                
                signals.append(signal)
                self.add_signal_to_history(signal)
                self.logger.info(f"Generated bearish signal: {signal}")
        
        return signals
    
    def filter_signals(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """
        Filter signals based on various criteria
        
        Args:
            signals: List of generated signals
            
        Returns:
            Filtered list of signals
        """
        filtered_signals = []
        
        for signal in signals:
            # Skip signals with low confidence
            if signal.confidence < self.confidence_threshold:
                continue
            
            # Check for duplicate signals (same type within a short time window)
            is_duplicate = False
            for hist_signal in self.signal_history[-10:]:  # Check last 10 signals
                time_diff = (signal.timestamp - hist_signal.timestamp).total_seconds()
                if time_diff < 300 and signal.signal_type == hist_signal.signal_type:  # 5 minutes
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_signals.append(signal)
        
        return filtered_signals