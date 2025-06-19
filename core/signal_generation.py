import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union

# Import position dataclass
from core.position import TradeSignal


class SignalGenerator:
    """
    Analyzes market data and generates trading signals based on candle patterns
    and technical indicators
    """
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """
        Initialize the signal generator
        
        Args:
            config: Configuration dictionary
            logger: Logger instance (optional)
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Load strategy parameters from config
        self.strategy_name = config.get('strategy_name', 'candle_pattern')
        self.confidence_threshold = float(config.get('confidence_threshold', 0.7))
        self.stop_loss_pct = float(config.get('stop_loss_pct', 0.3))
        self.take_profit_pct = float(config.get('take_profit_pct', 0.6))
        
        # Initialize signal history
        self.signal_history = []
        self.max_history_size = int(config.get('max_signal_history', 100))
    
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
    
    def analyze_candle_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze candle patterns in the price data
        
        Args:
            df: DataFrame with OHLCV price data
            
        Returns:
            Dictionary of detected patterns and their confidence scores
        """
        patterns = {}
        
        # Ensure we have enough data
        if len(df) < 5:
            self.logger.warning("Not enough data for pattern analysis")
            return patterns
        
        # Get the most recent candles
        recent_candles = df.iloc[-5:].copy()
        
        # Calculate candle properties
        recent_candles['body_size'] = abs(recent_candles['Close'] - recent_candles['Open'])
        recent_candles['upper_shadow'] = recent_candles['High'] - recent_candles[['Open', 'Close']].max(axis=1)
        recent_candles['lower_shadow'] = recent_candles[['Open', 'Close']].min(axis=1) - recent_candles['Low']
        recent_candles['range'] = recent_candles['High'] - recent_candles['Low']
        recent_candles['body_pct'] = recent_candles['body_size'] / recent_candles['range']
        recent_candles['is_bullish'] = recent_candles['Close'] > recent_candles['Open']
        
        # Get the last candle
        last_candle = recent_candles.iloc[-1]
        prev_candle = recent_candles.iloc[-2] if len(recent_candles) > 1 else None
        
        # Detect bullish engulfing pattern
        if prev_candle is not None and not prev_candle['is_bullish'] and last_candle['is_bullish']:
            if last_candle['Open'] <= prev_candle['Close'] and last_candle['Close'] >= prev_candle['Open']:
                confidence = min(1.0, last_candle['body_size'] / (prev_candle['body_size'] * 1.2))
                patterns['bullish_engulfing'] = confidence
        
        # Detect bearish engulfing pattern
        if prev_candle is not None and prev_candle['is_bullish'] and not last_candle['is_bullish']:
            if last_candle['Open'] >= prev_candle['Close'] and last_candle['Close'] <= prev_candle['Open']:
                confidence = min(1.0, last_candle['body_size'] / (prev_candle['body_size'] * 1.2))
                patterns['bearish_engulfing'] = confidence
        
        # Detect hammer pattern (bullish)
        if last_candle['lower_shadow'] >= 2 * last_candle['body_size'] and last_candle['upper_shadow'] < 0.5 * last_candle['body_size']:
            confidence = min(1.0, last_candle['lower_shadow'] / (last_candle['body_size'] * 3))
            patterns['hammer'] = confidence
        
        # Detect shooting star pattern (bearish)
        if last_candle['upper_shadow'] >= 2 * last_candle['body_size'] and last_candle['lower_shadow'] < 0.5 * last_candle['body_size']:
            confidence = min(1.0, last_candle['upper_shadow'] / (last_candle['body_size'] * 3))
            patterns['shooting_star'] = confidence
        
        # Detect doji pattern (indecision)
        if last_candle['body_size'] < 0.1 * last_candle['range']:
            confidence = 1.0 - (last_candle['body_size'] / (0.1 * last_candle['range']))
            patterns['doji'] = confidence
        
        # Detect morning star pattern (bullish)
        if len(recent_candles) >= 3:
            c1 = recent_candles.iloc[-3]
            c2 = recent_candles.iloc[-2]
            c3 = last_candle
            
            if not c1['is_bullish'] and c1['body_pct'] > 0.6 and \
               c2['body_pct'] < 0.3 and \
               c3['is_bullish'] and c3['body_pct'] > 0.6 and \
               c3['Close'] > (c1['Open'] + c1['Close']) / 2:
                confidence = min(1.0, c3['body_size'] / c1['body_size'])
                patterns['morning_star'] = confidence
        
        # Detect evening star pattern (bearish)
        if len(recent_candles) >= 3:
            c1 = recent_candles.iloc[-3]
            c2 = recent_candles.iloc[-2]
            c3 = last_candle
            
            if c1['is_bullish'] and c1['body_pct'] > 0.6 and \
               c2['body_pct'] < 0.3 and \
               not c3['is_bullish'] and c3['body_pct'] > 0.6 and \
               c3['Close'] < (c1['Open'] + c1['Close']) / 2:
                confidence = min(1.0, c3['body_size'] / c1['body_size'])
                patterns['evening_star'] = confidence
        
        return patterns
    
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
        patterns = self.analyze_candle_patterns(df)
        
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