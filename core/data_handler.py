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
from dotenv import load_dotenv
from enum import Enum
import random

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
        
        This class is responsible for fetching and processing market data. It
        takes a configuration dictionary, an optional DhanHQ API client
        instance, and an optional logger as parameters.
        
        Parameters
        ----------
        config : Dict[str, Any]
            The configuration dictionary. This contains the following keys:
            
            - 'data': This contains the following sub-keys:
                - 'cache_ttl_seconds': The number of seconds to cache data
                    before expiring it. Defaults to 60 seconds.
                - 'max_cache_size': The maximum number of cache entries to
                    store. Defaults to 1000.
                - 'throttle_ms': The number of milliseconds to throttle API
                    calls. Defaults to 200 ms.
        dhan_api : Optional
            An optional DhanHQ API client instance. If provided, this will be
            used to fetch real-time market data.
        logger : Optional
            An optional logger instance. If provided, this will be used to log
            messages.
        """
        self.config = config
        self.dhan_api = dhan_api
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize rate limiter with proper config
        throttle_ms = int(config.get('api', 'throttle_ms', fallback='200'))
        self.rate_limiter = RateLimiter(throttle_ms=throttle_ms)
        
        # Initialize cache
        self.cache = {}  # Stores the actual data
        self.cache_timestamps = {}  # Stores the timestamp of when each key was cached
        self.cache_ttl = int(config.get('data', 'cache_ttl_seconds', fallback='60'))
        self.max_cache_size = int(config.get('data', 'max_cache_size', fallback='1000'))
        
        # Load environment variables from dhan_credentials.env
        load_dotenv('.env')
        
        # Initialize Dhan API if not provided
        if self.dhan_api is None and DHANHQ_AVAILABLE:
            client_id = os.getenv('DHAN_CLIENT_ID')
            access_token = os.getenv('DHAN_ACCESS_TOKEN')
            if client_id and access_token:
                self.dhan_api = dhanhq(client_id, access_token)
            else:
                self.logger.warning("Dhan API credentials missing in env file")
        
    
    def _clean_cache(self) -> None:
        """
        Clean expired cache entries

        This method is responsible for cleaning out expired cache entries. 

        It does the following:

        1. Iterate over all cached keys and check if they are expired
        2. If a key is expired, add it to a list of expired keys
        3. Remove all expired keys from the cache
        4. If the cache is too large, sort all keys by timestamp (oldest first)
        5. Remove the oldest entries until the cache is the correct size

        Args:
            None

        Returns:
            None
        """
        current_time = time_lib.time()
        expired_keys = []
        
        # Iterate over all cached keys and check if they are expired
        for key, timestamp in self.cache_timestamps.items():
            if current_time - timestamp > self.cache_ttl:
                # If a key is expired, add it to a list of expired keys
                expired_keys.append(key)
        
        # Remove all expired keys from the cache
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

    def _validate_api_response(self, response: Any) -> bool:
        """
        Validate API response structure

        This method checks if the API response is valid by verifying the following conditions:
        1. The response is not empty or None.
        2. The response is a dictionary.
        3. The 'status' key in the response dictionary is equal to 'success'.
        4. The 'data' key exists in the response dictionary and is not empty.

        If any of the above conditions are not met, the method returns False indicating an invalid response.
        """
        if not response or not isinstance(response, dict):
            self.logger.debug("API response is not a dictionary or is None")
            return False
        
        if response.get('status') != 'success':
            error_msg = response.get('message', 'Unknown error')
            self.logger.debug(f"API response status is not success: {error_msg}")
            return False
            
        # Additional validation for data structure
        if 'data' not in response:
            self.logger.debug("API response missing 'data' key")
            return False
            
        return True
        # If all the above checks pass, returns True indicating a valid response.

    @rate_limited(limiter=lambda self: self.rate_limiter, key="get_nifty_spot")
    def get_nifty_spot(self) -> float:
        """
        Get the current NIFTY spot price
        
        This function fetches the current NIFTY spot price from the Dhan API if
        available, or falls back to a simulated price if not.

        The function first checks if the result is already cached. If it is, it
        returns the cached value. If not, it fetches the price from the Dhan API
        and caches the result. If the Dhan API is unavailable, it generates a
        simulated price and caches that instead.

        Returns:
            Current NIFTY spot price
        """
        
        cache_key = "nifty_spot"
        self._clean_cache()
        
        # Return cached value if available
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        try:
            if self.dhan_api and DHANHQ_AVAILABLE:
                # Get NIFTY spot price from Dhan API
                response = self.dhan_api.intraday_minute_data(
                    security_id="13", 
                    exchange_segment="IDX_I", 
                    instrument_type="INDEX", 
                    from_date="2025-06-19", 
                    to_date="2025-06-19"
                )
                
                if response and 'data' in response:
                    # Extract the spot price from the response
                    spot_price = float(response['data']['close'][-1])
                    
                    # Cache the result
                    self.cache[cache_key] = spot_price
                    self.cache_timestamps[cache_key] = time_lib.time()
                    
                    # Return the spot price
                    return spot_price
                else:
                    # Log an error if the Dhan API returned an unexpected response
                    self.logger.error(f"Failed to get NIFTY spot price: {response}")
            
            # Fallback to simulation mode if Dhan API unavailable
            self.logger.warning("Using simulated NIFTY spot price")
            
            # Generate a realistic NIFTY price (around 18000-19000)
            simulated_price = 18500 + (np.random.random() * 500)
            
            # Cache the result
            self.cache[cache_key] = simulated_price
            self.cache_timestamps[cache_key] = time_lib.time()
            
            # Return the simulated spot price
            return simulated_price
            
        except Exception as e:
            # Log an error if an exception was raised
            self.logger.error(f"Error getting NIFTY spot price: {e}")
            # Return a default value in case of error
            return 18500.0
            
    @rate_limited(limiter=lambda self: self.rate_limiter, key="get_recent_nifty_data")
    def get_recent_nifty_data(self, interval_minutes: int = 1, days: int = 1) -> pd.DataFrame:
        """
        Get recent NIFTY price data
        
        This function fetches recent NIFTY price data from the DhanHQ API if
        available, or falls back to simulated data if not.

        The function first checks if the result is already cached. If it is, it
        returns the cached value. If not, it fetches the price data from the
        DhanHQ API and caches the result. If the DhanHQ API is unavailable, it
        generates simulated price data and caches that instead.

        Args:
            interval_minutes: Candle interval in minutes
            days: Number of days of historical data to fetch
            
        Returns:
            DataFrame with NIFTY price data
        """
        # Generate a unique cache key based on the interval and days
        cache_key = f"nifty_data_{interval_minutes}_{days}"
        
        # Clean expired cache entries
        self._clean_cache()
        
        # Check if data is present in cache and return it if available
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        try:
            # Check if DhanHQ API is available and instantiated
            if self.dhan_api and DHANHQ_AVAILABLE:
                # Calculate the date range for the historical data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Format dates as strings for API request
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")
                
                # Fetch historical data from DhanHQ API
                response = self.dhan_api.intraday_minute_data(
                    security_id="13",
                    exchange_segment="IDX_I",
                    instrument_type="INDEX", 
                    from_date=start_str, 
                    to_date=end_str
                )
                
                # Check if response contains data
                if response and 'data' in response:
                    # Convert response data to a DataFrame
                    df = pd.DataFrame(response['data'])
                    
                    # Convert 'timestamp' column to datetime type and sort the DataFrame
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    
                    # Rename columns to match standard OHLCV format
                    # Ensure we use lowercase column names as expected by signal generator
                    df = df.rename(columns={
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'volume': 'volume',
                        'timestamp': 'date'
                    })
                    
                    # Set 'date' as the DataFrame index
                    df = df.set_index('date')
                    
                    # Cache the DataFrame result
                    self.cache[cache_key] = df
                    self.cache_timestamps[cache_key] = time_lib.time()
                    
                    return df
                else:
                    # Log an error if the response does not contain expected data
                    self.logger.error(f"Failed to get NIFTY historical data: {response}")
            
            # If DhanHQ API is not available, simulate data
            self.logger.warning("Using simulated NIFTY historical data")
            
            # Generate simulated date range based on interval and days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Create a date range with specified interval
            date_range = pd.date_range(
                start=start_date,
                end=end_date,
                freq=f"{interval_minutes}min"
            )
            
            # Initialize parameters for simulated price generation
            base_price = 18500
            price_volatility = 50
            
            # Prepare a list to hold simulated OHLCV data
            data = []
            prev_close = base_price
            
            # Iterate over each datetime in the date range
            for dt in date_range:
                # Skip times outside market hours (9:15 AM to 3:30 PM)
                if dt.time() < time(9, 15) or dt.time() > time(15, 30):
                    continue
                
                # Skip weekends
                if dt.weekday() > 4:  # 5 = Saturday, 6 = Sunday
                    continue
                
                # Simulate OHLCV (Open, High, Low, Close, Volume) data
                change = np.random.normal(0, price_volatility)
                close = prev_close + change
                high = close + abs(np.random.normal(0, price_volatility/2))
                low = close - abs(np.random.normal(0, price_volatility/2))
                open_price = prev_close + np.random.normal(0, price_volatility/4)
                volume = int(np.random.normal(1000000, 500000))
                
                # Append the simulated data to the list
                data.append({
                    'date': dt,
                    'open': max(open_price, low),
                    'high': max(high, open_price, close),
                    'low': min(low, open_price, close),
                    'close': close,
                    'volume': max(0, volume)
                })
                
                # Update previous close with current close for next iteration
                prev_close = close
            
            # Convert the list of simulated data to a DataFrame
            df = pd.DataFrame(data)
            df = df.set_index('date')
            
            # Cache the simulated DataFrame result
            self.cache[cache_key] = df
            self.cache_timestamps[cache_key] = time_lib.time()
            
            return df
            
        except Exception as e:
            # Log any exception that occurs during data fetching
            self.logger.error(f"Error getting NIFTY historical data: {e}")
            # Return an empty DataFrame in case of an error
            return pd.DataFrame()
            
    @rate_limited(limiter=lambda self: self.rate_limiter, key="get_option_chain")
    def get_option_chain(self, expiry_date: date) -> Optional[pd.DataFrame]:
        """
        Get NIFTY option chain data

        This method retrieves the option chain data for NIFTY by following these steps:
        1. Checks if the Dhan API is available and connects to it if necessary.
        2. If the Dhan API is available, calls the option chain method to retrieve the option chain data.
        3. If the API call is successful, parses the response into a pandas DataFrame.
        4. Standardizes column names in the DataFrame.
        5. Returns the retrieved option chain data, or None if the data could not be retrieved.

        Args:
            expiry_date: The expiry date for the option chain.

        Returns:
            Optional[pd.DataFrame]: The retrieved option chain data, or None if the data could not be retrieved.
        """
        try:
            # Step 1: Check if the Dhan API is available and connect to it if necessary
            if not self.dhan_api and DHANHQ_AVAILABLE:
                client_id = os.getenv('DHAN_CLIENT_ID')
                access_token = os.getenv('DHAN_ACCESS_TOKEN')
                if client_id and access_token:
                    self.dhan_api = dhanhq(client_id, access_token)

            expiry_str = expiry_date.strftime("%Y-%m-%d")
            cache_key = f"option_chain_{expiry_str}"
            self._clean_cache()
            
                # Return cached value if available
            if cache_key in self.cache:
                return self.cache[cache_key]
                
                # Call the option chain method
            chain_response = self.dhan_api.option_chain(
                under_security_id=13,  # NIFTY ID
                under_exchange_segment=ExchangeSegment.INDEX.value,
                expiry=expiry_str
            )

                # Step 3: If the API call is successful, parse the response
            if self._validate_api_response(chain_response):
                data = chain_response.get('data', {}).get('data', {}).get('oc', {})
                if not data:
                    self.logger.error("No option chain data in response")
                    return None
                
                    # Flatten data into DataFrame
                rows = []
                for strike, strike_data in data.items():
                    strike_float = f"{float(strike):.6f}"
                    for option_type, option_data in strike_data.items():
                        rows.append({
                            'strike': strike_float,
                            'type': option_type.strip().lower(),
                            'last_price': option_data.get('last_price', 0),
                            'volume': option_data.get('total_traded_volume', 0),
                            'oi': option_data.get('open_interest', 0),
                            'bid_price': option_data.get('bid_price', 0),
                            'ask_price': option_data.get('ask_price', 0),
                            'implied_volatility': option_data.get('implied_volatility', 0)
                        })
                
                if not rows:
                    self.logger.error("No option data found after parsing")
                    return None
                
                    # Create and process DataFrame
                df = pd.DataFrame(rows)
                numeric_cols = ['last_price', 'volume', 'oi', 'bid_price', 'ask_price', 'implied_volatility']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df.fillna(0, inplace=True)
                
                    # Cache the DataFrame result
                self.cache[cache_key] = df
                self.cache_timestamps[cache_key] = time_lib.time()
                
                return df
            else:
                self.logger.error("Invalid API response for option chain")
                return None

        except Exception as e:
            self.logger.error(f"Error getting option chain: {e}")
            return None
    
    def get_current_and_next_expiry(self, ref_date: Union[str, pd.Timestamp]) -> Tuple[date, date]:
        """
        Determine the current and next expiry dates based on a given reference date.

        Args:
            ref_date (str or pd.Timestamp): The reference date from which expiry dates are calculated.

        Returns:
            Tuple[date, date]: A tuple containing the current and next expiry dates.
        """
        # Convert the reference date to a date object if it's in string format
        if isinstance(ref_date, str):
            ref_date = pd.to_datetime(ref_date).date()
        # Convert the reference date to a date object if it's a timestamp
        elif isinstance(ref_date, pd.Timestamp):
            ref_date = ref_date.date()

        # Get expiry day from config (default to Thursday = 3)
        self.expiry_day = int(self.config.get('data', 'weekly_expiry_day', fallback='3'))

        # This logic correctly finds the next upcoming expiry date.
        # It calculates the number of days ahead to reach the expiry day
        # (e.g. Thursday) and then adds that number of days to the reference date.
        days_ahead = (self.expiry_day - ref_date.weekday() + 7) % 7
        # If today is Thursday (days_ahead=0), we want next Thursday (7 days ahead)
        if days_ahead == 0:
            days_ahead = 7
        # Determine the current expiry date by adding the calculated days to the reference date
        current_expiry = ref_date + timedelta(days=days_ahead)
        # Determine the next expiry date, which is one week after the current expiry
        next_expiry = current_expiry + timedelta(weeks=1)

        # Store the calculated current expiry date for later use
        self.current_expiry = current_expiry
        # Store the calculated next expiry date for later use
        self.next_expiry = next_expiry
        # Format the current expiry date as a string for logging purposes
        self.expiry_string = self.current_expiry.strftime('%Y-%m-%d')
        # Log the calculated expiry date for reference
        self.logger.info(f"Trading expiry date set to: {self.expiry_string}")

        # Return the tuple of current and next expiry dates
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
            itm_offset = int(self.config.get('data', 'itm_strike_offset', fallback='2'))
            
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
            options_data = option_chain[option_chain['type'] == 'ce']
        else:  # PE
            options_data = option_chain[option_chain['type'] == 'pe']
        
        # Get specific strike data
        strike_data = options_data[options_data['strike'] == strike]
        
        if not strike_data.empty:
            # Add symbol and type information
            strike_data['symbol'] = self.get_option_symbol(strike, option_type, expiry_date)
            strike_data['type'] = option_type
            strike_data['expiryDate'] = expiry_date.strftime("%Y-%m-%d")
            
            return strike_data.to_dict(orient='records')[0]
        else:
            # If strike not found, generate simulated data
            self.logger.warning(f"Strike {strike} {option_type} not found in option chain, using simulated data")
            
            # Get spot price
            spot_price = self.get_nifty_spot()
            
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
                'last_price': round(price, 1),
                'change': round(np.random.normal(0, price * 0.03), 1),
                'volume': int(1000 * np.random.random()),
                'oi': int(5000 * np.random.random()),
                'bid_price': round(price * 0.98, 1),
                'ask_price': round(price * 1.02, 1),
                'implied_volatility': round(volatility * (0.8 + 0.4 * np.random.random()), 2),
                'symbol': self.get_option_symbol(strike, option_type, expiry_date),
                'type': option_type,
                'expiryDate': expiry_date.strftime("%Y-%m-%d")
            }
        
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
        return option_details.get('last_price', 0.0)

    def get_market_data(self) -> Tuple[pd.DataFrame, float]:
        """
        Fetch market data including OHLC candles and current spot price
        
        Args:
            None
        
        Returns:
            Tuple of (DataFrame with market data, current spot price)
        """
        try:
            # Get recent NIFTY data
            interval_minutes = int(self.config.get('data', 'ohlc_timeframe', fallback='5'))
            days = int(self.config.get('data', 'max_history_days', fallback='1'))
            
            df = self.get_recent_nifty_data(interval_minutes, days)
            
            # Get current spot price
            spot_price = self.get_nifty_spot()
            
            return df, spot_price
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            raise

class ExchangeSegment(Enum):
    """Enum for exchange segments"""
    EQUITY = "EQUITY"
    DERIVATIVE = "FNO"
    CURRENCY = "CDS"
    COMMODITY = "COMM"
    INDEX = "IDX_I"
