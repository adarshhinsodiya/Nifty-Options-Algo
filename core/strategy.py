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
from core.data_handler import DataHandler
from core.signal_generation import SignalGenerator
from core.execution import ExecutionHandler
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
    
    def __init__(self, config_path: Optional[str] = None):
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
        log_level = self.config.get('logging', 'level', fallback='INFO')
        log_to_file = self.config.getboolean('logging', 'log_to_file', fallback=False)
        max_log_size = self.config.get('logging', 'max_log_size_mb', fallback='10')
        backup_count = self.config.get('logging', 'backup_count', fallback='3')
        log_dir = self.config.get('logging', 'log_dir', fallback='logs')

        # Create logger config dict
        logger_config = {
            'level': log_level,
            'log_to_file': log_to_file,
            'max_log_size_mb': max_log_size,
            'backup_count': backup_count
        }

        self.logger = setup_logger('nifty_options', logger_config, log_dir)
        
        # Log configuration
        self.logger.info(f"Loaded configuration from {config_path if config_path else 'default'}")

        # Get mode
        self.mode = 'simulation' if self.config.get('mode', 'live', fallback='false').lower() == 'false' else self.config.get('mode', 'live', fallback='simulation')
        self.logger.info(f"Running in {self.mode} mode")
        
        # Initialize Dhan API client if available
        self.dhan_api = None
        if DHANHQ_AVAILABLE and self.mode.lower() == 'live':
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