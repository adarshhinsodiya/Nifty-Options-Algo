import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, date
import pytz
import os
import logging
from typing import Optional, Dict, Any, Tuple, List, Union
import time as time_lib
import json
from collections import defaultdict

# Import utilities
from utils.rate_limit import RateLimiter, rate_limited

# Check if dhanhq is available
try:
    from dhanhq import dhanhq
    DHANHQ_AVAILABLE = True
except ImportError:
    DHANHQ_AVAILABLE = False
    print("Warning: dhanhq library not found. Running in simulation mode.")


class DataHandler:
    """
    Handles market data fetching and processing
    """
    
    def __init__(self, config: Dict[str, Any], dhan_api=None, logger=None):
        """
        Initialize the data handler
        
        Args:
            config: Configuration dictionary
            dhan_api: DhanHQ API client instance (optional)
            logger: Logger instance (optional)
        """
        self.config = config
        self.dhan_api = dhan_api
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize cache
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = int(self.config.get('data', 'cache_ttl_seconds', fallback='60'))
        self.max_cache_size = int(self.config.get('data', 'max_cache_size', fallback='1000'))
        
        # Initialize rate limiter
        throttle_ms = int(self.config.get('data', 'throttle_ms', fallback='200'))
        self.rate_limiter = RateLimiter(throttle_ms=throttle_ms)
    
    def _clean_cache(self) -> None:
        """
        Clean expired cache entries
        """
        current_time = time_lib.time()
        expired_keys = []
        
        # Find expired keys
        for key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(key)
        
        # Remove expired keys
        for key in expired_keys:
            if key in self.cache:
                del self.cache[key]
            if key in self.cache_timestamps:
                del self.cache_timestamps[key]
        
        # Limit cache size if needed
        if len(self.cache) > self.max_cache_size:
            # Sort keys by timestamp (oldest first)
            sorted_keys = sorted(
                self.cache_timestamps.keys(),
                key=lambda k: self.cache_timestamps[k]
            )
            
            # Remove oldest entries
            keys_to_remove = sorted_keys[:len(self.cache) - self.max_cache_size]
            for key in keys_to_remove:
                if key in self.cache:
                    del self.cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
                    
    @rate_limited(limiter=None, key="get_nifty_spot")
    def get_nifty_spot(self) -> float:
        """
        Get the current NIFTY spot price
        
        Returns:
            Current NIFTY spot price
        """
        # Use the instance rate limiter if none provided in decorator
        if not hasattr(self.get_nifty_spot, "_rate_limiter"):
            self.get_nifty_spot._rate_limiter = self.rate_limiter
        
        cache_key = "nifty_spot"
        self._clean_cache()
        
        # Return cached value if available
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            if self.dhan_api and DHANHQ_AVAILABLE:
                # Get NIFTY spot price from Dhan API
                response = self.dhan_api.get_quote("NIFTY", "IDX_I")
                
                if response and 'data' in response:
                    spot_price = float(response['data']['ltp'])
                    
                    # Cache the result
                    self.cache[cache_key] = spot_price
                    self.cache_timestamps[cache_key] = time_lib.time()
                    
                    return spot_price
                else:
                    self.logger.error(f"Failed to get NIFTY spot price: {response}")
            
            # Fallback to simulation mode
            self.logger.warning("Using simulated NIFTY spot price")
            # Generate a realistic NIFTY price (around 18000-19000)
            simulated_price = 18500 + (np.random.random() * 500)
            
            # Cache the result
            self.cache[cache_key] = simulated_price
            self.cache_timestamps[cache_key] = time_lib.time()
            
            return simulated_price
            
        except Exception as e:
            self.logger.error(f"Error getting NIFTY spot price: {e}")
            # Return a default value in case of error
            return 18500.0
    
    @rate_limited(limiter=None, key="get_recent_nifty_data")
    def get_recent_nifty_data(self, interval_minutes: int = 5, days: int = 1) -> pd.DataFrame:
        """
        Get recent NIFTY price data
        
        Args:
            interval_minutes: Candle interval in minutes
            days: Number of days of historical data to fetch
            
        Returns:
            DataFrame with NIFTY price data
        """
        # Use the instance rate limiter if none provided in decorator
        if not hasattr(self.get_recent_nifty_data, "_rate_limiter"):
            self.get_recent_nifty_data._rate_limiter = self.rate_limiter
        
        cache_key = f"nifty_data_{interval_minutes}_{days}"
        self._clean_cache()
        
        # Return cached value if available
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            if self.dhan_api and DHANHQ_AVAILABLE:
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Format dates for API
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")
                
                # Get historical data from Dhan API
                response = self.dhan_api.get_historical_data(
                    "NIFTY", 
                    "IDX_I", 
                    start_str, 
                    end_str, 
                    f"{interval_minutes}m"
                )
                
                if response and 'data' in response:
                    # Convert to DataFrame
                    df = pd.DataFrame(response['data'])
                    
                    # Process DataFrame
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    
                    # Rename columns to standard OHLCV format
                    df = df.rename(columns={
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume',
                        'timestamp': 'Date'
                    })
                    
                    # Set index to timestamp
                    df = df.set_index('Date')
                    
                    # Cache the result
                    self.cache[cache_key] = df
                    self.cache_timestamps[cache_key] = time_lib.time()
                    
                    return df
                else:
                    self.logger.error(f"Failed to get NIFTY historical data: {response}")
            
            # Fallback to simulation mode
            self.logger.warning("Using simulated NIFTY historical data")
            
            # Generate simulated data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Create date range
            date_range = pd.date_range(
                start=start_date,
                end=end_date,
                freq=f"{interval_minutes}min"
            )
            
            # Generate random walk prices
            base_price = 18500
            price_volatility = 50
            
            # Generate OHLCV data
            data = []
            prev_close = base_price
            
            for dt in date_range:
                # Skip non-market hours (9:15 AM to 3:30 PM)
                if dt.time() < time(9, 15) or dt.time() > time(15, 30):
                    continue
                
                # Skip weekends
                if dt.weekday() > 4:  # 5 = Saturday, 6 = Sunday
                    continue
                
                # Generate candle data
                change = np.random.normal(0, price_volatility)
                close = prev_close + change
                high = close + abs(np.random.normal(0, price_volatility/2))
                low = close - abs(np.random.normal(0, price_volatility/2))
                open_price = prev_close + np.random.normal(0, price_volatility/4)
                volume = int(np.random.normal(1000000, 500000))
                
                data.append({
                    'Date': dt,
                    'Open': max(open_price, low),
                    'High': max(high, open_price, close),
                    'Low': min(low, open_price, close),
                    'Close': close,
                    'Volume': max(0, volume)
                })
                
                prev_close = close
            
            # Create DataFrame
            df = pd.DataFrame(data)
            df = df.set_index('Date')
            
            # Cache the result
            self.cache[cache_key] = df
            self.cache_timestamps[cache_key] = time_lib.time()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting NIFTY historical data: {e}")
            # Return an empty DataFrame in case of error
            return pd.DataFrame()
    
    @rate_limited(limiter=None, key="get_option_chain")
    def get_option_chain(self, expiry_date: date) -> Dict[str, Any]:
        """
        Get NIFTY option chain for a specific expiry date
        
        Args:
            expiry_date: Expiry date for the options
            
        Returns:
            Dictionary with option chain data
        """
        # Use the instance rate limiter if none provided in decorator
        if not hasattr(self.get_option_chain, "_rate_limiter"):
            self.get_option_chain._rate_limiter = self.rate_limiter
        
        expiry_str = expiry_date.strftime("%Y-%m-%d")
        cache_key = f"option_chain_{expiry_str}"
        self._clean_cache()
        
        # Return cached value if available
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            if self.dhan_api and DHANHQ_AVAILABLE:
                # Get option chain from Dhan API
                response = self.dhan_api.get_option_chain("NIFTY", expiry_str)
                
                if response and 'data' in response:
                    option_chain = response['data']
                    
                    # Cache the result
                    self.cache[cache_key] = option_chain
                    self.cache_timestamps[cache_key] = time_lib.time()
                    
                    return option_chain
                else:
                    self.logger.error(f"Failed to get option chain: {response}")
            
            # Fallback to simulation mode
            self.logger.warning("Using simulated option chain data")
            
            # Get current NIFTY spot price
            spot_price = self.get_nifty_spot()
            
            # Round to nearest 50
            atm_strike = round(spot_price / 50) * 50
            
            # Generate strikes (±1000 points from ATM)
            strikes = range(atm_strike - 1000, atm_strike + 1050, 50)
            
            # Generate option chain
            option_chain = {
                'spotPrice': spot_price,
                'timestamp': datetime.now().isoformat(),
                'expiryDate': expiry_str,
                'calls': {},
                'puts': {}
            }
            
            for strike in strikes:
                # Calculate theoretical prices based on Black-Scholes approximation
                days_to_expiry = (expiry_date - date.today()).days
                if days_to_expiry < 1:
                    days_to_expiry = 1
                
                time_to_expiry = days_to_expiry / 365.0
                volatility = 0.2  # 20% annualized volatility
                
                # Simplified pricing model
                call_itm = max(0, spot_price - strike)
                put_itm = max(0, strike - spot_price)
                
                # Time value approximation
                time_value = spot_price * volatility * np.sqrt(time_to_expiry)
                
                # Final prices
                call_price = call_itm + (time_value * np.exp(-0.5 * ((strike - spot_price) / (spot_price * volatility))**2))
                put_price = put_itm + (time_value * np.exp(-0.5 * ((spot_price - strike) / (spot_price * volatility))**2))
                
                # Generate volume and open interest
                base_volume = max(100, int(10000 * np.exp(-0.5 * ((strike - spot_price) / 200)**2)))
                base_oi = max(500, int(50000 * np.exp(-0.5 * ((strike - spot_price) / 300)**2)))
                
                # Add random variation
                call_volume = int(base_volume * (0.8 + 0.4 * np.random.random()))
                put_volume = int(base_volume * (0.8 + 0.4 * np.random.random()))
                call_oi = int(base_oi * (0.8 + 0.4 * np.random.random()))
                put_oi = int(base_oi * (0.8 + 0.4 * np.random.random()))
                
                # Create option data
                option_chain['calls'][strike] = {
                    'strike': strike,
                    'lastPrice': round(call_price, 1),
                    'change': round(np.random.normal(0, call_price * 0.03), 1),
                    'volume': call_volume,
                    'openInterest': call_oi,
                    'bidPrice': round(call_price * 0.98, 1),
                    'askPrice': round(call_price * 1.02, 1),
                    'impliedVolatility': round(volatility * (0.8 + 0.4 * np.random.random()), 2)
                }
                
                option_chain['puts'][strike] = {
                    'strike': strike,
                    'lastPrice': round(put_price, 1),
                    'change': round(np.random.normal(0, put_price * 0.03), 1),
                    'volume': put_volume,
                    'openInterest': put_oi,
                    'bidPrice': round(put_price * 0.98, 1),
                    'askPrice': round(put_price * 1.02, 1),
                    'impliedVolatility': round(volatility * (0.8 + 0.4 * np.random.random()), 2)
                }
            
            # Cache the result
            self.cache[cache_key] = option_chain
            self.cache_timestamps[cache_key] = time_lib.time()
            
            return option_chain
            
        except Exception as e:
            self.logger.error(f"Error getting option chain: {e}")
            # Return an empty dictionary in case of error
            return {'calls': {}, 'puts': {}, 'spotPrice': self.get_nifty_spot()}
    
    def get_current_and_next_expiry(self, ref_date: Union[str, pd.Timestamp]) -> Tuple[date, date]:
        """
        Determine the current and next expiry dates based on a given reference date
        
        Args:
            ref_date: Reference date from which expiry dates are calculated
            
        Returns:
            Tuple containing the current and next expiry dates
        """
        # Convert reference date to datetime if it's a string
        if isinstance(ref_date, str):
            ref_date = pd.Timestamp(ref_date)
        
        # Convert to date if it's a timestamp
        if isinstance(ref_date, pd.Timestamp):
            ref_date = ref_date.date()
        
        # Get expiry selection from config
        expiry_selection = self.config.get('data', 'expiry_selection', fallback='weekly').lower()
        
        if expiry_selection == 'weekly':
            # Weekly expiry (usually Thursday)
            weekly_expiry_day = int(self.config.get('data', 'weekly_expiry_day', fallback='3'))  # Thursday = 3
            
            # Find the next expiry day (next Thursday)
            days_to_add = (weekly_expiry_day - ref_date.weekday()) % 7
            if days_to_add == 0:  # If today is Thursday
                # If it's past market hours, use next week
                current_time = datetime.now().time()
                if current_time > time(15, 30):  # Past 3:30 PM
                    days_to_add = 7
            
            current_expiry = ref_date + timedelta(days=days_to_add)
            next_expiry = current_expiry + timedelta(days=7)
            
        else:  # Monthly expiry
            # Monthly expiry (last Thursday of the month)
            monthly_expiry_day = int(self.config.get('data', 'monthly_expiry_day', fallback='25'))
            
            # Get current month's expiry
            current_month = ref_date.replace(day=1)
            next_month = (current_month + timedelta(days=32)).replace(day=1)
            
            # Find last Thursday of current month
            last_day = (next_month - timedelta(days=1)).day
            current_expiry = None
            
            for day in range(min(monthly_expiry_day, last_day), 0, -1):
                test_date = ref_date.replace(day=day)
                if test_date.weekday() == 3:  # Thursday
                    current_expiry = test_date
                    break
            
            # If we've passed this month's expiry, use next month
            if current_expiry < ref_date:
                current_month = next_month
                next_month = (current_month + timedelta(days=32)).replace(day=1)
                
                # Find last Thursday of next month
                last_day = (next_month - timedelta(days=1)).day
                for day in range(min(monthly_expiry_day, last_day), 0, -1):
                    test_date = current_month.replace(day=day)
                    if test_date.weekday() == 3:  # Thursday
                        current_expiry = test_date
                        break
            
            # Find last Thursday of the month after next
            last_day = ((next_month + timedelta(days=32)).replace(day=1) - timedelta(days=1)).day
            next_expiry = None
            
            for day in range(min(monthly_expiry_day, last_day), 0, -1):
                test_date = next_month.replace(day=day)
                if test_date.weekday() == 3:  # Thursday
                    next_expiry = test_date
                    break
        
        return current_expiry, next_expiry
    
    def select_option_strike(self, option_type: str, spot_price: float, expiry_date: date) -> int:
        """
        Select an appropriate option strike based on strategy settings
        
        Args:
            option_type: 'CE' for call options, 'PE' for put options
            spot_price: Current NIFTY spot price
            expiry_date: Option expiry date
            
        Returns:
            Selected strike price
        """
        # Get strike selection method from config
        strike_selection = self.config.get('data', 'strike_selection', fallback='atm').lower()
        
        # Round spot price to nearest 50
        atm_strike = round(spot_price / 50) * 50
        
        if strike_selection == 'atm':
            # At-the-money
            return atm_strike
        
        elif strike_selection == 'otm':
            # Out-of-the-money
            otm_offset = int(self.config.get('data', 'otm_strike_offset', fallback='1'))
            
            if option_type == 'CE':
                # For calls, OTM is above spot
                return atm_strike + (otm_offset * 50)
            else:  # PE
                # For puts, OTM is below spot
                return atm_strike - (otm_offset * 50)
        
        elif strike_selection == 'itm':
            # In-the-money
            itm_offset = int(self.config.get('data', 'itm_strike_offset', fallback='1'))
            
            if option_type == 'CE':
                # For calls, ITM is below spot
                return atm_strike - (itm_offset * 50)
            else:  # PE
                # For puts, ITM is above spot
                return atm_strike + (itm_offset * 50)
        
        else:
            # Default to ATM if invalid selection
            self.logger.warning(f"Invalid strike selection '{strike_selection}', using ATM")
            return atm_strike
    
    def get_option_symbol(self, strike: int, option_type: str, expiry_date: date) -> str:
        """
        Generate the option symbol for a given strike, type, and expiry
        
        Args:
            strike: Strike price
            option_type: 'CE' for call options, 'PE' for put options
            expiry_date: Option expiry date
            
        Returns:
            Option symbol string
        """
        # Format expiry date as required by the exchange (e.g., 29JUN23)
        month_map = {
            1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN',
            7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'
        }
        
        expiry_str = f"{expiry_date.day}{month_map[expiry_date.month]}{expiry_date.year % 100}"
        
        # Format: NIFTY29JUN2318500CE
        symbol = f"NIFTY{expiry_str}{strike}{option_type}"
        
        return symbol
    
    def get_option_details(self, strike: int, option_type: str, expiry_date: date) -> Dict[str, Any]:
        """
        Get detailed information for a specific option
        
        Args:
            strike: Strike price
            option_type: 'CE' for call options, 'PE' for put options
            expiry_date: Option expiry date
            
        Returns:
            Dictionary with option details
        """
        # Get option chain
        option_chain = self.get_option_chain(expiry_date)
        
        # Get option data based on type
        if option_type == 'CE':
            options_data = option_chain.get('calls', {})
        else:  # PE
            options_data = option_chain.get('puts', {})
        
        # Get specific strike data
        strike_data = options_data.get(strike, {})
        
        if not strike_data:
            # If strike not found, generate simulated data
            self.logger.warning(f"Strike {strike} {option_type} not found in option chain, using simulated data")
            
            # Get spot price
            spot_price = option_chain.get('spotPrice', self.get_nifty_spot())
            
            # Calculate days to expiry
            days_to_expiry = max(1, (expiry_date - date.today()).days)
            time_to_expiry = days_to_expiry / 365.0
            volatility = 0.2  # 20% annualized volatility
            
            # Simplified pricing model
            if option_type == 'CE':
                intrinsic = max(0, spot_price - strike)
            else:  # PE
                intrinsic = max(0, strike - spot_price)
            
            # Time value approximation
            time_value = spot_price * volatility * np.sqrt(time_to_expiry)
            price_factor = np.exp(-0.5 * ((strike - spot_price) / (spot_price * volatility))**2)
            price = intrinsic + (time_value * price_factor)
            
            # Create simulated data
            strike_data = {
                'strike': strike,
                'lastPrice': round(price, 1),
                'change': round(np.random.normal(0, price * 0.03), 1),
                'volume': int(1000 * np.random.random()),
                'openInterest': int(5000 * np.random.random()),
                'bidPrice': round(price * 0.98, 1),
                'askPrice': round(price * 1.02, 1),
                'impliedVolatility': round(volatility * (0.8 + 0.4 * np.random.random()), 2)
            }
        
        # Add symbol and type information
        strike_data['symbol'] = self.get_option_symbol(strike, option_type, expiry_date)
        strike_data['type'] = option_type
        strike_data['expiryDate'] = expiry_date.strftime("%Y-%m-%d")
        
        return strike_data
    
    def get_option_quote(self, strike: int, option_type: str, expiry_date: date) -> float:
        """
        Get the current price of a specific option
        
        Args:
            strike: Strike price
            option_type: 'CE' for call options, 'PE' for put options
            expiry_date: Option expiry date
            
        Returns:
            Current option price
        """
        option_details = self.get_option_details(strike, option_type, expiry_date)
        return option_details.get('lastPrice', 0.0)
