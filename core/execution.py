import pandas as pd
import numpy as np
import logging
import uuid
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import position dataclasses and enums
from core.position import Position, TradeSignal, OrderType, TransactionType, ProductType, ExchangeSegment

# Check if breeze_connect is available
try:
    from breeze_connect import BreezeConnect
    BREEZE_CONNECT_AVAILABLE = True
except ImportError:
    BREEZE_CONNECT_AVAILABLE = False
    print("Warning: breeze_connect library not found. Running in simulation mode.")


class ExecutionHandler:
    """
    Handles trade execution, position management, and P&L calculations
    """
    
    def __init__(self, config: Dict[str, Any], data_handler=None, breeze_api=None, logger=None):
        """
        Initialize execution handler with Breeze Connect API
        
        Args:
            config: Configuration dictionary
            data_handler: Instance of DataHandler for market data
            breeze_api: Optional pre-initialized BreezeConnect instance
            logger: Optional logger instance
        """
        self.config = config
        self.data_handler = data_handler
        self.logger = logger or logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv()
        
        # Initialize Breeze Connect API if not provided
        if breeze_api is None and BREEZE_CONNECT_AVAILABLE:
            api_key = os.getenv('ICICI_API_KEY')
            api_secret = os.getenv('ICICI_API_SECRET')
            session_token = os.getenv('ICICI_SESSION_TOKEN')
            user_id = os.getenv('ICICI_USER_ID')
            
            if all([api_key, api_secret, session_token]):
                self.logger.debug("Initializing Breeze Connect API...")
                try:
                    # Configure session with retry and keep-alive
                    session = requests.Session()
                    retry_strategy = Retry(
                        total=3,
                        backoff_factor=1,
                        status_forcelist=[500, 502, 503, 504]
                    )
                    adapter = HTTPAdapter(max_retries=retry_strategy)
                    session.mount("https://", adapter)
                    session.headers.update({'Connection': 'keep-alive'})
                    
                    # Initialize BreezeConnect with custom session
                    breeze_api = BreezeConnect(api_key=api_key)
                    breeze_api.generate_session(api_secret=api_secret, session_token=session_token)
                    
                    # Set user ID if provided
                    if user_id:
                        breeze_api.user_id = user_id
                    
                    # Set timeout
                    breeze_api._session.timeout = 30  # 30 second timeout
                    
                    self.logger.info("Breeze Connect API initialized successfully")
                except Exception as e:
                    self.logger.error(f"Failed to initialize Breeze Connect API: {str(e)}")
                    if "session" in locals():
                        session.close()
                    raise
            else:
                self.logger.error("Breeze Connect API credentials missing in env file")
        
        self.breeze_api = breeze_api
        
        # Load execution parameters from config
        self.mode = config.get('mode', 'simulation').lower()
        self.capital = float(config.get('capital', 'initial_capital', fallback='100000'))
        self.risk_percent = float(config.get('capital', 'risk_percent', fallback='0.01'))
        self.max_positions = int(config.get('capital', 'max_positions', fallback='5'))
        self.position_size_pct = float(config.get('strategy', 'position_size_pct', fallback='0.1'))
        self.slippage_pct = float(config.get('strategy', 'slippage_pct', fallback='0.01'))
        
        # Initialize positions list
        self.active_positions = []
        self.closed_positions = []
        
        # Initialize P&L tracking
        self.total_pnl = 0.0
        self.daily_pnl = {}
    
    def execute_signal(self, signal: TradeSignal) -> Optional[Position]:
        """
        Execute a trade signal
        
        Args:
            signal: Trade signal to execute
            
        Returns:
            Position object if trade executed, None otherwise
        """
        # Check if we can take more positions
        if len(self.active_positions) >= self.max_positions:
            self.logger.warning(f"Max positions ({self.max_positions}) reached, skipping signal")
            return None
        
        # Calculate position size
        position_size = self.calculate_position_size(signal)
        
        # Determine strike price if not specified in signal
        if signal.strike == 0:
            strike = self.data_handler.select_option_strike(
                signal.option_type, 
                signal.spot_price, 
                datetime.strptime(signal.option_data.get('expiryDate', ''), '%Y-%m-%d').date() 
                if signal.option_data and 'expiryDate' in signal.option_data 
                else self.data_handler.get_current_and_next_expiry(datetime.now())[0]
            )
        else:
            strike = signal.strike
        
        # Get current expiry date
        expiry_date = self.data_handler.get_current_and_next_expiry(datetime.now())[0]
        
        # Get option details
        option_details = self.data_handler.get_option_details(strike, signal.option_type, expiry_date)
        
        # Determine entry price with slippage
        entry_price = option_details.get('lastPrice', 0.0)
        if signal.signal_type == "BUY_CALL" or signal.signal_type == "BUY_PUT":
            # Add slippage for buy orders
            entry_price *= (1 + self.slippage_pct)
        else:
            # Subtract slippage for sell orders
            entry_price *= (1 - self.slippage_pct)
        
        # Round entry price to nearest 0.05
        entry_price = round(entry_price * 20) / 20
        
        # Calculate quantity based on position size and entry price
        quantity = max(1, int(position_size / entry_price))
        
        # Create position ID
        position_id = str(uuid.uuid4())[:8]
        
        # Execute the trade
        order_id = None
        if self.mode == 'live' and self.breeze_api and BREEZE_CONNECT_AVAILABLE:
            try:
                # Determine buy/sell action
                action = "buy" if signal.signal_type.startswith("BUY") else "sell"
                
                # Get option symbol for Breeze Connect
                option_symbol = self.data_handler.get_option_symbol(
                    strike, 
                    signal.option_type, 
                    expiry_date
                )
                
                # Place order via Breeze Connect API
                response = self.breeze_api.place_order(
                    stock_code=option_symbol,
                    exchange_code="NFO",
                    product=ProductType.INTRADAY.value,
                    action=action.upper(),
                    order_type=OrderType.MARKET.value,
                    quantity=str(quantity),
                    price="0",  # Market order
                    validity="day",
                    validity_date="",  # Current day
                    disclosed_quantity="0",
                    expiry_date=expiry_date.strftime("%Y-%m-%d"),
                    right=signal.option_type.upper(),
                    strike_price=str(strike),
                    user_remark=f"AlgoTrade_{signal.signal_type}"
                )
                
                if response and isinstance(response, dict) and 'Success' in response and response['Success']:
                    order_id = response.get('reference_id', str(uuid.uuid4()))
                    self.logger.info(f"Order placed successfully: {order_id}")
                else:
                    error_msg = response.get('Error', 'Unknown error') if isinstance(response, dict) else str(response)
                    self.logger.error(f"Failed to place order: {error_msg}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"Error placing order: {e}", exc_info=True)
                return None
        else:
            # Simulation mode
            order_id = f"SIM_{str(uuid.uuid4())[:8]}"
            self.logger.info(f"Simulated order {order_id} for {quantity} {option_details.get('symbol', '')} at {entry_price}")
        
        # Create position object
        position = Position(
            id=position_id,
            order_id=order_id,
            type=signal.signal_type,
            strike=strike,
            option_type=signal.option_type,
            quantity=quantity,
            entry_price=entry_price,
            entry_timestamp=datetime.now(),
            status="ACTIVE",
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        # Add to active positions
        self.active_positions.append(position)
        
        self.logger.info(f"Position opened: {position}")
        return position
    
    def calculate_position_size(self, signal: TradeSignal) -> float:
        """
        Calculate the position size for a trade
        
        Args:
            signal: Trade signal
            
        Returns:
            Position size in rupees
        """
        # Base position size as percentage of capital
        base_size = self.capital * self.position_size_pct
        
        # Adjust based on signal confidence
        adjusted_size = base_size * signal.confidence
        
        # Ensure minimum position size
        min_size = self.capital * 0.02  # Minimum 2% of capital
        position_size = max(min_size, adjusted_size)
        
        # Ensure maximum position size
        max_size = self.capital * 0.25  # Maximum 25% of capital
        position_size = min(max_size, position_size)
        
        return position_size
    
    def update_positions(self) -> None:
        """
        Update all active positions with current market data
        and check for exit conditions
        """
        for position in self.active_positions[:]:  # Create a copy to iterate
            # Get current option price
            expiry_date = self.data_handler.get_current_and_next_expiry(datetime.now())[0]
            current_price = self.data_handler.get_option_quote(
                position.strike, 
                position.option_type, 
                expiry_date
            )
            
            # Get current spot price
            spot_price = self.data_handler.get_nifty_spot()
            
            # Check exit conditions
            exit_reason = None
            
            # Stop loss hit
            if position.stop_loss is not None:
                if (position.type.startswith("BUY") and spot_price >= position.stop_loss) or \
                   (position.type.startswith("SELL") and spot_price <= position.stop_loss):
                    exit_reason = "STOP_LOSS"
            
            # Take profit hit
            if position.take_profit is not None:
                if (position.type.startswith("BUY") and spot_price <= position.take_profit) or \
                   (position.type.startswith("SELL") and spot_price >= position.take_profit):
                    exit_reason = "TAKE_PROFIT"
            
            # Time-based exit (end of day)
            current_time = datetime.now().time()
            if current_time.hour >= 15 and current_time.minute >= 15:  # 3:15 PM
                exit_reason = "EOD"
            
            # Exit position if conditions met
            if exit_reason:
                self.close_position(position, current_price, exit_reason)
    
    def close_position(self, position: Position, exit_price: float = None, exit_reason: str = "MANUAL") -> None:
        """
        Close an active position
        
        Args:
            position: Position to close
            exit_price: Exit price (optional, will fetch current price if not provided)
            exit_reason: Reason for closing the position
        """
        # Get current price if not provided
        if exit_price is None:
            expiry_date = self.data_handler.get_current_and_next_expiry(datetime.now())[0]
            exit_price = self.data_handler.get_option_quote(
                position.strike, 
                position.option_type, 
                expiry_date
            )
        
        # Apply slippage to exit price
        if position.type.startswith("BUY"):
            # Subtract slippage for sell orders (closing a buy position)
            exit_price *= (1 - self.slippage_pct)
        else:
            # Add slippage for buy orders (closing a sell position)
            exit_price *= (1 + self.slippage_pct)
        
        # Round exit price to nearest 0.05
        exit_price = round(exit_price * 20) / 20
        
        # Execute the trade
        if self.mode == 'live' and self.breeze_api and BREEZE_CONNECT_AVAILABLE:
            try:
                # Determine buy/sell action (opposite of position type)
                action = "sell" if position.type.startswith("BUY") else "buy"
                
                # Get option symbol for Breeze Connect
                expiry_date = self.data_handler.get_current_and_next_expiry(datetime.now())[0]
                option_symbol = self.data_handler.get_option_symbol(
                    position.strike, 
                    position.option_type, 
                    expiry_date
                )
                
                # Place close order via Breeze Connect API
                response = self.breeze_api.place_order(
                    stock_code=option_symbol,
                    exchange_code="NFO",
                    product=ProductType.INTRADAY.value,
                    action=action.upper(),
                    order_type=OrderType.MARKET.value,
                    quantity=str(position.quantity),
                    price="0",  # Market order
                    validity="day",
                    validity_date="",  # Current day
                    disclosed_quantity="0",
                    expiry_date=expiry_date.strftime("%Y-%m-%d"),
                    right=position.option_type.upper(),
                    strike_price=str(position.strike),
                    user_remark=f"AlgoTrade_CLOSE_{position.id}"
                )
                
                if response and isinstance(response, dict) and 'Success' in response and response['Success']:
                    order_id = response.get('reference_id', str(uuid.uuid4()))
                    self.logger.info(f"Close order placed successfully: {order_id}")
                else:
                    error_msg = response.get('Error', 'Unknown error') if isinstance(response, dict) else str(response)
                    self.logger.error(f"Failed to place close order: {error_msg}")
                    
            except Exception as e:
                self.logger.error(f"Error placing close order: {e}", exc_info=True)
        else:
            # Simulation mode
            order_id = f"SIM_CLOSE_{str(uuid.uuid4())[:8]}"
            self.logger.info(f"Simulated close order {order_id} for position {position.id} at {exit_price}")
        
        # Calculate P&L
        pnl = (exit_price - position.entry_price) * position.quantity
        if position.type.startswith("SELL"):
            pnl = -pnl  # Reverse for short positions
        
        # Calculate return percentage
        position_value = position.entry_price * position.quantity
        return_pct = (pnl / position_value) * 100 if position_value > 0 else 0
        
        # Update position
        position.exit_price = exit_price
        position.exit_timestamp = datetime.now()
        position.exit_reason = exit_reason
        position.pnl = pnl
        position.return_pct = return_pct
        position.status = "CLOSED"
        
        # Move from active to closed positions
        self.active_positions.remove(position)
        self.closed_positions.append(position)
        
        # Update total P&L
        self.total_pnl += pnl
        
        # Update daily P&L
        today = datetime.now().date().isoformat()
        if today not in self.daily_pnl:
            self.daily_pnl[today] = 0
        self.daily_pnl[today] += pnl
        
        self.logger.info(f"Position closed: {position.id}, P&L: {pnl:.2f}, Return: {return_pct:.2f}%")
    
    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all positions and P&L
        
        Returns:
            Dictionary with position and P&L summary
        """
        active_count = len(self.active_positions)
        closed_count = len(self.closed_positions)
        
        # Calculate today's P&L
        today = datetime.now().date().isoformat()
        today_pnl = self.daily_pnl.get(today, 0)
        
        # Calculate unrealized P&L
        unrealized_pnl = 0
        for position in self.active_positions:
            expiry_date = self.data_handler.get_current_and_next_expiry(datetime.now())[0]
            current_price = self.data_handler.get_option_quote(
                position.strike, 
                position.option_type, 
                expiry_date
            )
            
            position_pnl = (current_price - position.entry_price) * position.quantity
            if position.type.startswith("SELL"):
                position_pnl = -position_pnl  # Reverse for short positions
            
            unrealized_pnl += position_pnl
        
        # Calculate total P&L (realized + unrealized)
        total_pnl = self.total_pnl + unrealized_pnl
        
        # Calculate return on capital
        return_pct = (total_pnl / self.capital) * 100 if self.capital > 0 else 0
        
        return {
            "active_positions": active_count,
            "closed_positions": closed_count,
            "realized_pnl": self.total_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "return_pct": return_pct,
            "today_pnl": today_pnl
        }
    
    def export_positions_to_csv(self, filepath: str) -> None:
        """
        Export all positions to a CSV file
        
        Args:
            filepath: Path to the output CSV file
        """
        # Combine active and closed positions
        all_positions = self.active_positions + self.closed_positions
        
        # Convert to DataFrame
        positions_data = []
        for pos in all_positions:
            pos_dict = {
                "id": pos.id,
                "order_id": pos.order_id,
                "type": pos.type,
                "strike": pos.strike,
                "option_type": pos.option_type,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "entry_timestamp": pos.entry_timestamp,
                "status": pos.status,
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
                "exit_price": pos.exit_price,
                "exit_timestamp": pos.exit_timestamp,
                "exit_reason": pos.exit_reason,
                "pnl": pos.pnl,
                "return_pct": pos.return_pct
            }
            positions_data.append(pos_dict)
        
        # Create DataFrame and export
        if positions_data:
            df = pd.DataFrame(positions_data)
            df.to_csv(filepath, index=False)
            self.logger.info(f"Positions exported to {filepath}")
        else:
            self.logger.warning("No positions to export")
