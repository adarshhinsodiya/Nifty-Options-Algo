import os
import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import asdict
import logging
from dotenv import load_dotenv
import json
import time

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now imports will work in all cases
from .signal_generation import SignalGenerator
from .data_handler import DataHandler
from .execution import ExecutionHandler
from core.position import Position, TradeSignal
from utils.config_loader import ConfigLoader
from utils.logger import setup_logger
from utils.rate_limit import rate_limited

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
    
    def __init__(self, config_path: Optional[str] = None, mode: str = "simulation"):
        """
        Initialize the strategy
        
        Args:
            config_path: Path to the configuration file (optional)
            mode: Running mode ("simulation" or "live")
        """
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        
        # Properly set mode in config
        if not self.config.has_section('mode'):
            self.config.add_section('mode')
        self.config.set('mode', 'live', 'true' if mode.lower() == 'live' else 'false')
        self.config.set('mode', 'simulation', 'true' if mode.lower() == 'simulation' else 'false')
        
        # Setup logger
        log_level = self.config.get('logging', 'level', fallback='INFO')
        log_to_file = self.config.getboolean('logging', 'log_to_file', fallback=False)
        self.logger = setup_logger('nifty_options', log_level, log_to_file)
        
        # Get mode
        self.mode = 'live' if self.config.getboolean('mode', 'live', fallback=False) else 'simulation'
        self.logger.info(f"Running in {self.mode} mode")
        
        # Initialize Dhan API client if available
        self.dhan_api = None
        if DHANHQ_AVAILABLE and self.mode.lower() == 'live':
            try:
                client_id = os.environ.get('DHAN_CLIENT_ID')
                access_token = os.environ.get('DHAN_ACCESS_TOKEN')
                
                if client_id and access_token:
                    self.dhan_api = dhanhq(client_id, access_token)
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
        self.last_data_fetch_time = datetime.now() - timedelta(days=1)  # Force first fetch
        self.last_summary_time = datetime.now() - timedelta(minutes=5)
        self.df = None
        self.spot_price = None
        
        # Initialize performance tracking
        self.start_time = None
        self.end_time = None
        self.iteration_count = 0
        self.signal_count = 0
        self.trade_count = 0
    
    def start(self) -> None:
        """
        Start the strategy execution
        """
        self.running = True
        self.logger.info(f"Strategy started at {datetime.now()}")
        
        try:
            # Initial market data fetch - critical for strategy operation
            try:
                self._fetch_market_data(max_retries=5)  # More retries for initial fetch
                self.logger.info("Successfully fetched initial market data")
            except Exception as e:
                self.logger.critical(f"Failed to get initial market data: {e}")
                raise
                
            # Main strategy loop
            interval_seconds = int(self.config.get('config', 'iteration_interval_seconds', fallback='60'))
            
            while self.running:
                if not self._is_market_open():
                    self.logger.info("Market is closed, waiting...")
                    time.sleep(60)  # Check every minute
                    continue
                
                try:
                    self._run_iteration()
                except Exception as e:
                    self.logger.error(f"Error in strategy iteration: {e}")
                    if not self.running:  # Don't sleep if we're shutting down
                        raise
                    time.sleep(30)  # Wait before retrying
                    
                time.sleep(interval_seconds)
                
        except Exception as e:
            self.logger.error(f"Error in strategy: {e}", exc_info=True)
            raise
        finally:
            self.stop()
    
    def stop(self) -> None:
        """
        Stop the strategy and save state
        """
        self.running = False
        self.logger.info("Strategy stopping...")
        
        try:
            # Ensure data directory exists
            data_dir = self.config.get('data', 'data_dir', fallback='data')
            os.makedirs(data_dir, exist_ok=True)
            
            # Save strategy state
            state_file = os.path.join(data_dir, 'strategy_state.json')
            with open(state_file, 'w') as f:
                json.dump(self.get_state(), f)
                
            self.logger.info(f"Strategy state saved to {state_file}")
        except Exception as e:
            self.logger.error(f"Error saving strategy state: {e}")
    
    def _run_iteration(self) -> None:
        """
        Run one iteration of the strategy
        """
        try:
            # Fetch market data with retry logic
            self._fetch_market_data()
            
            # Generate signals with both dataframe and spot price
            signals = self.signal_generator.generate_signals(self.df, self.spot_price)
            
            if signals:
                self.logger.info(f"Generated {len(signals)} signals")
                for signal in signals:
                    self.execution_handler.execute_signal(signal)
            
        except Exception as e:
            self.logger.error(f"Error in strategy iteration: {e}")
            raise
    
    def _fetch_market_data(self, max_retries: int = 3) -> None:
        """
        Fetch market data (OHLC and spot price) from data handler
        
        Args:
            max_retries: Maximum number of retry attempts for API failures
        """
        retry_count = 0
        while retry_count < max_retries:
            try:
                self.df, self.spot_price = self.data_handler.get_market_data()
                if self.df.empty:
                    raise ValueError("Received empty market data")
                return
                
            except Exception as e:
                retry_count += 1
                wait_time = min(60, 5 * retry_count)  # Exponential backoff with max 60s
                self.logger.warning(
                    f"Failed to fetch market data (attempt {retry_count}/{max_retries}): {e}"
                )
                
                if retry_count < max_retries:
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    self.logger.error("Max retries reached for market data fetch")
                    raise
    
    def _is_market_open(self) -> bool:
        """
        Check if market is currently open based on configured hours
        
        Returns:
            True if market is open, False otherwise
        """
        try:
            # Get current time
            now = datetime.now().time()
            
            # Parse market hours with fallbacks
            open_time = datetime.strptime(
                self.config.get('market', 'market_open_time', fallback='09:15'), 
                '%H:%M'
            ).time()
            
            close_time = datetime.strptime(
                self.config.get('market', 'market_close_time', fallback='15:30'), 
                '%H:%M'
            ).time()
            
            # Check if current time is within market hours
            return open_time <= now <= close_time
            
        except Exception as e:
            self.logger.error(f"Error checking market hours: {e}")
            return False
    
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
            "mode": self.mode,
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
    
    def get_state(self) -> dict:
        """
        Get current strategy state for saving
        
        Returns:
            Dictionary containing strategy state
        """
        return {
            "positions": [pos.__dict__ for pos in self.execution_handler.active_positions],
            "iteration_count": self.iteration_count,
            "signal_count": self.signal_count,
            "trade_count": self.trade_count,
            "last_updated": datetime.now().isoformat()
        }