import pandas as pd
import numpy as np
import logging
import uuid
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union

# Import position dataclasses and enums
from core.position import Position, TradeSignal, OrderType, TransactionType, ProductType, ExchangeSegment

# Check if dhanhq is available
try:
    from dhanhq import dhanhq
    DHANHQ_AVAILABLE = True
except ImportError:
    DHANHQ_AVAILABLE = False
    print("Warning: dhanhq library not found. Running in simulation mode.")


class ExecutionHandler:
    """
    Handles trade execution, position management, and P&L calculations
    """
    
    def __init__(self, config: Dict[str, Any], data_handler=None, dhan_api=None, logger=None):
        """
        Initialize the execution handler
        
        Args:
            config: Configuration dictionary
            data_handler: DataHandler instance for market data
            dhan_api: DhanHQ API client instance (optional)
            logger: Logger instance (optional)
        """
        self.config = config
        self.data_handler = data_handler
        self.dhan_api = dhan_api
        self.logger = logger or logging.getLogger(__name__)
        
        # Load execution parameters from config
        self.mode = config.get('mode', 'simulation').lower()
        self.capital = float(config.get('capital', 100000))
        self.max_positions = int(config.get('max_positions', 2))
        self.position_size_pct = float(config.get('position_size_pct', 0.1))
        self.slippage_pct = float(config.get('slippage_pct', 0.01))
        
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
        if self.mode == 'live' and self.dhan_api and DHANHQ_AVAILABLE:
            try:
                # Determine transaction type
                transaction_type = TransactionType.BUY if signal.signal_type.startswith("BUY") else TransactionType.SELL
                
                # Place order via Dhan API
                response = self.dhan_api.place_order(
                    security_id=option_details.get('symbol', ''),
                    exchange_segment=ExchangeSegment.NFO.value,
                    transaction_type=transaction_type.value,
                    quantity=quantity,
                    order_type=OrderType.MARKET.value,
                    product_type=ProductType.INTRADAY.value,
                    price=0.0  # Market order
                )
                
                if response and 'data' in response and 'orderId' in response['data']:
                    order_id = response['data']['orderId']
                    self.logger.info(f"Order placed successfully: {order_id}")
                else:
                    self.logger.error(f"Failed to place order: {response}")
                    return None
                    
            except Exception as e:
                self.logger.error(f"Error placing order: {e}")
                return None
        else:
            # Simulation mode
            self.logger.info(f"Simulated order for {quantity} {option_details.get('symbol', '')} at {entry_price}")
        
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
        if self.mode == 'live' and self.dhan_api and DHANHQ_AVAILABLE:
            try:
                # Determine transaction type (opposite of position type)
                transaction_type = TransactionType.SELL if position.type.startswith("BUY") else TransactionType.BUY
                
                # Get option symbol
                expiry_date = self.data_handler.get_current_and_next_expiry(datetime.now())[0]
                option_symbol = self.data_handler.get_option_symbol(
                    position.strike, 
                    position.option_type, 
                    expiry_date
                )
                
                # Place order via Dhan API
                response = self.dhan_api.place_order(
                    security_id=option_symbol,
                    exchange_segment=ExchangeSegment.NFO.value,
                    transaction_type=transaction_type.value,
                    quantity=position.quantity,
                    order_type=OrderType.MARKET.value,
                    product_type=ProductType.INTRADAY.value,
                    price=0.0  # Market order
                )
                
                if response and 'data' in response and 'orderId' in response['data']:
                    self.logger.info(f"Close order placed successfully: {response['data']['orderId']}")
                else:
                    self.logger.error(f"Failed to place close order: {response}")
                    
            except Exception as e:
                self.logger.error(f"Error placing close order: {e}")
        else:
            # Simulation mode
            self.logger.info(f"Simulated close order for position {position.id} at {exit_price}")
        
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
