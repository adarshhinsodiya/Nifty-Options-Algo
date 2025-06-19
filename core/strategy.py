import pandas as pd
import numpy as np
import logging
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dotenv import load_dotenv

# Import core modules
from core.data_handler import DataHandler
from core.signal_generation import SignalGenerator
from core.execution import ExecutionHandler
from core.position import Position, TradeSignal

# Import utilities
from utils.logger import setup_logger
from utils.config_loader import ConfigLoader

# Check if dhanhq is available
try:
    from dhanhq import dhanhq
    DHANHQ_AVAILABLE = True
except ImportError:
    DHANHQ_AVAILABLE = False
    print("Warning: dhanhq library not found. Running in simulation mode.")


class NiftyOptionsStrategy:
    """
    Main strategy class that orchestrates data handling, signal generation, and execution
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the strategy
        
        Args:
            config_path: Path to the configuration file (optional)
        """
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        
        # Setup logger
        log_level = self.config.get('log_level', 'INFO')
        log_dir = self.config.get('log_dir', 'logs')
        self.logger = setup_logger(log_level, log_dir)
        
        # Log configuration
        self.logger.info(f"Loaded configuration from {config_path if config_path else 'default'}")
        self.logger.info(f"Running in {self.config.get('mode', 'simulation')} mode")
        
        # Initialize Dhan API client if available
        self.dhan_api = None
        if DHANHQ_AVAILABLE and self.config.get('mode', 'simulation').lower() == 'live':
            try:
                client_id = os.environ.get('DHAN_CLIENT_ID')
                access_token = os.environ.get('DHAN_ACCESS_TOKEN')
                
                if client_id and access_token:
                    self.dhan_api = dhanhq.Client(client_id, access_token)
                    self.logger.info("DhanHQ API client initialized successfully")
                else:
                    self.logger.warning("Missing DhanHQ API credentials, running in simulation mode")
            except Exception as e:
                self.logger.error(f"Error initializing DhanHQ API client: {e}")
        
        # Initialize components
        self.data_handler = DataHandler(self.config, self.dhan_api, self.logger)
        self.signal_generator = SignalGenerator(self.config, self.logger)
        self.execution_handler = ExecutionHandler(self.config, self.data_handler, self.dhan_api, self.logger)
        
        # Initialize strategy state
        self.running = False
        self.last_signal_time = datetime.now() - timedelta(hours=1)
        self.last_data_fetch_time = datetime.now() - timedelta(minutes=5)
        self.last_summary_time = datetime.now() - timedelta(minutes=5)
        
        # Initialize performance tracking
        self.start_time = None
        self.end_time = None
        self.iteration_count = 0
        self.signal_count = 0
        self.trade_count = 0
    
    def start(self) -> None:
        """
        Start the strategy
        """
        self.running = True
        self.start_time = datetime.now()
        self.logger.info(f"Strategy started at {self.start_time}")
        
        try:
            # Main strategy loop
            while self.running:
                self.iteration_count += 1
                
                # Check if market is open
                if not self._is_market_open():
                    self.logger.info("Market is closed, waiting...")
                    time.sleep(60)  # Check every minute
                    continue
                
                # Run one iteration of the strategy
                self._run_iteration()
                
                # Sleep for the specified interval
                interval_seconds = int(self.config.get('check_interval_seconds', 60))
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            self.logger.info("Strategy interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in strategy: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self) -> None:
        """
        Stop the strategy and perform cleanup
        """
        self.running = False
        self.end_time = datetime.now()
        
        # Close all open positions
        self._close_all_positions()
        
        # Export positions to CSV
        data_dir = self.config.get('data_dir', 'data')
        os.makedirs(data_dir, exist_ok=True)
        positions_file = os.path.join(data_dir, f"positions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.execution_handler.export_positions_to_csv(positions_file)
        
        # Log summary
        duration = (self.end_time - self.start_time).total_seconds() / 60 if self.start_time else 0
        summary = self.execution_handler.get_position_summary()
        
        self.logger.info(f"Strategy stopped at {self.end_time}")
        self.logger.info(f"Duration: {duration:.2f} minutes")
        self.logger.info(f"Iterations: {self.iteration_count}")
        self.logger.info(f"Signals generated: {self.signal_count}")
        self.logger.info(f"Trades executed: {self.trade_count}")
        self.logger.info(f"P&L summary: {summary}")
    
    def _run_iteration(self) -> None:
        """
        Run one iteration of the strategy
        """
        current_time = datetime.now()
        
        # Fetch market data
        if (current_time - self.last_data_fetch_time).total_seconds() >= int(self.config.get('data_fetch_interval_seconds', 60)):
            self._fetch_market_data()
            self.last_data_fetch_time = current_time
        
        # Update existing positions
        self.execution_handler.update_positions()
        
        # Generate signals
        if (current_time - self.last_signal_time).total_seconds() >= int(self.config.get('signal_interval_seconds', 300)):
            self._generate_and_execute_signals()
            self.last_signal_time = current_time
        
        # Log position summary
        if (current_time - self.last_summary_time).total_seconds() >= int(self.config.get('summary_interval_seconds', 300)):
            summary = self.execution_handler.get_position_summary()
            self.logger.info(f"Position summary: {summary}")
            self.last_summary_time = current_time
    
    def _fetch_market_data(self) -> None:
        """
        Fetch and update market data
        """
        try:
            # Get NIFTY spot price
            spot_price = self.data_handler.get_nifty_spot()
            self.logger.debug(f"NIFTY spot price: {spot_price}")
            
            # Get recent NIFTY data
            interval_minutes = int(self.config.get('candle_interval_minutes', 5))
            days = int(self.config.get('historical_days', 1))
            
            df = self.data_handler.get_recent_nifty_data(interval_minutes, days)
            self.logger.debug(f"Fetched {len(df)} candles of NIFTY data")
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
    
    def _generate_and_execute_signals(self) -> None:
        """
        Generate signals and execute trades
        """
        try:
            # Get recent NIFTY data
            interval_minutes = int(self.config.get('candle_interval_minutes', 5))
            days = int(self.config.get('historical_days', 1))
            df = self.data_handler.get_recent_nifty_data(interval_minutes, days)
            
            if df.empty:
                self.logger.warning("No data available for signal generation")
                return
            
            # Get current spot price
            spot_price = self.data_handler.get_nifty_spot()
            
            # Generate signals
            signals = self.signal_generator.generate_signals(df, spot_price)
            
            # Filter signals
            filtered_signals = self.signal_generator.filter_signals(signals)
            
            self.signal_count += len(filtered_signals)
            
            # Execute signals
            for signal in filtered_signals:
                position = self.execution_handler.execute_signal(signal)
                if position:
                    self.trade_count += 1
            
        except Exception as e:
            self.logger.error(f"Error generating or executing signals: {e}")
    
    def _close_all_positions(self) -> None:
        """
        Close all open positions
        """
        for position in self.execution_handler.active_positions[:]:
            self.logger.info(f"Closing position {position.id} on strategy stop")
            self.execution_handler.close_position(position, exit_reason="STRATEGY_STOP")
    
    def _is_market_open(self) -> bool:
        """
        Check if the market is currently open
        
        Returns:
            True if market is open, False otherwise
        """
        # Get current time in India timezone
        now = datetime.now()
        
        # Check if it's a weekday (0=Monday, 4=Friday)
        if now.weekday() > 4:  # Weekend
            return False
        
        # Check market hours (9:15 AM to 3:30 PM)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        # Allow pre-market preparation
        pre_market = int(self.config.get('pre_market_minutes', 15))
        market_open = market_open - timedelta(minutes=pre_market)
        
        # Allow post-market cleanup
        post_market = int(self.config.get('post_market_minutes', 15))
        market_close = market_close + timedelta(minutes=post_market)
        
        return market_open <= now <= market_close
    
    def backtest(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Run a backtest of the strategy
        
        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            
        Returns:
            Dictionary with backtest results
        """
        self.logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # TODO: Implement backtesting functionality
        self.logger.warning("Backtesting not yet implemented")
        
        return {
            "status": "not_implemented",
            "message": "Backtesting functionality is not yet implemented"
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the strategy
        
        Returns:
            Dictionary with strategy status
        """
        # Get position summary
        position_summary = self.execution_handler.get_position_summary()
        
        # Calculate runtime
        runtime = (datetime.now() - self.start_time).total_seconds() / 60 if self.start_time else 0
        
        return {
            "running": self.running,
            "mode": self.config.get('mode', 'simulation'),
            "runtime_minutes": runtime,
            "iterations": self.iteration_count,
            "signals_generated": self.signal_count,
            "trades_executed": self.trade_count,
            "active_positions": position_summary["active_positions"],
            "closed_positions": position_summary["closed_positions"],
            "realized_pnl": position_summary["realized_pnl"],
            "unrealized_pnl": position_summary["unrealized_pnl"],
            "total_pnl": position_summary["total_pnl"],
            "return_pct": position_summary["return_pct"],
            "today_pnl": position_summary["today_pnl"],
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }