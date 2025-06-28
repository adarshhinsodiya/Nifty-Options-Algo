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

# Check if breeze_connect is available
try:
    from breeze_connect import BreezeConnect
    BREEZE_CONNECT_AVAILABLE = True
except ImportError:
    BREEZE_CONNECT_AVAILABLE = False
    print("Warning: breeze_connect library not found. API functionality will not be available.")


class DataHandler:
    """
    Handles market data fetching and processing
    """
    
    def __init__(self, config: Dict[str, Any], breeze_api=None, logger=None):
        """
        Initialize the data handler
        
        This class is responsible for fetching and processing market data. It
        takes a configuration dictionary, an optional Breeze Connect API client
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
        breeze_api : Optional
            An optional Breeze Connect API client instance. If provided, this will be
            used to fetch real-time market data.
        logger : Optional
            An optional logger instance. If provided, this will be used to log
            messages.
        """
        self.config = config
        self.breeze_api = breeze_api
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize rate limiter with proper config
        throttle_ms = int(config.get('api', 'throttle_ms', fallback='200'))
        self.rate_limiter = RateLimiter(throttle_ms=throttle_ms)
        
        # Initialize cache
        self.cache = {}  # Stores the actual data
        self.cache_timestamps = {}  # Stores the timestamp of when each key was cached
        self.cache_ttl = int(config.get('data', 'cache_ttl_seconds', fallback='60'))
        self.max_cache_size = int(config.get('data', 'max_cache_size', fallback='1000'))
        
        # Load environment variables
        load_dotenv('.env')
        
        # Initialize Breeze Connect API if not provided
        if self.breeze_api is None and BREEZE_CONNECT_AVAILABLE:
            api_key = os.getenv('ICICI_API_KEY')
            api_secret = os.getenv('ICICI_API_SECRET')
            session_token = os.getenv('ICICI_SESSION_TOKEN')
            user_id = os.getenv('ICICI_USER_ID')
            
            self.logger.debug(f"Environment variables loaded - API key exists: {bool(api_key)}")
            
            if api_key and api_secret and session_token:
                self.logger.debug("Initializing Breeze Connect API...")
                try:
                    self.breeze_api = BreezeConnect(api_key=api_key)
                    self.breeze_api.generate_session(api_secret=api_secret, session_token=session_token)
                    if user_id:
                        self.breeze_api.user_id = user_id
                    
                    # Test API connection
                    test_response = self.breeze_api.get_funds()
                    self.logger.debug(f"API test response: {test_response}")
                    self.logger.debug("Breeze Connect API initialized and tested successfully")
                except Exception as e:
                    self.logger.error(f"Failed to initialize Breeze Connect API: {str(e)}")
                    raise
            else:
                self.logger.error("Breeze Connect API credentials missing in env file")
        
    
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
        Validate API response structure with detailed error messages
        
        Args:
            response: The API response to validate
            
        Returns:
            bool: True if the response is valid, False otherwise
        """
        if not response:
            self.logger.error("API returned empty response")
            return False
            
        # Breeze Connect may return a list for some endpoints
        if isinstance(response, list):
            if not response:
                self.logger.error("API returned empty list")
                return False
            return True
            
        # For dictionary responses
        if isinstance(response, dict):
            # Check for error in response
            if 'Error' in response and response['Error'] is not None:
                self.logger.error(f"API returned error: {response['Error']}")
                return False
                
            # Check for success status
            if 'Success' in response and not response['Success']:
                self.logger.error("API returned unsuccessful status")
                return False
                
            # For historical data responses
            if isinstance(response.get('Success'), list):
                return True
                
            # For other successful responses
            return True
            
        self.logger.error(f"Unexpected API response type: {type(response)}")
        return False

    @rate_limited(limiter=lambda self: self.rate_limiter, key="get_nifty_spot")
    def get_nifty_spot(self) -> float:
        """
        Get the current NIFTY spot price
        
        This function fetches the current NIFTY spot price from the Breeze Connect API.
        
        Returns:
            float: The current NIFTY spot price
            
        Raises:
            ValueError: If the API call fails or returns invalid data
        """
        try:
            if not self.breeze_api and BREEZE_CONNECT_AVAILABLE:
                api_key = os.getenv('ICICI_API_KEY')
                api_secret = os.getenv('ICICI_API_SECRET')
                session_token = os.getenv('ICICI_SESSION_TOKEN')
                
                if api_key and api_secret and session_token:
                    self.breeze_api = BreezeConnect(api_key=api_key)
                    self.breeze_api.generate_session(api_secret=api_secret, session_token=session_token)
            
            cache_key = "nifty_spot"
            self._clean_cache()
        
            # Return cached value if available
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Get current date in required format
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Get NIFTY index data
            response = self.breeze_api.get_historical_data(
                interval="1minute",
                from_date=today,
                to_date=today,
                stock_code="NIFTY",
                exchange_code="NSE",
                product_type="cash"
            )
            
            if self._validate_api_response(response):
                # For Breeze Connect, response is a list of dictionaries
                if isinstance(response, list) and len(response) > 0:
                    # Get the latest data point (most recent)
                    latest_data = response[-1]
                    latest_close = float(latest_data.get('close', 0))
                    
                    if latest_close > 0:
                        self.cache[cache_key] = latest_close
                        self.cache_timestamps[cache_key] = time_lib.time()
                        return latest_close
                
            # Fallback to using get_quotes if historical data fails
            quote_response = self.breeze_api.get_quotes(
                stock_code="NIFTY",
                exchange_code="NSE",
                expiry_date="",
                product_type="cash",
                right="",
                strike_price=""
            )
            
            if self._validate_api_response(quote_response):
                latest_close = float(quote_response.get('last', 0))
                if latest_close > 0:
                    self.cache[cache_key] = latest_close
                    self.cache_timestamps[cache_key] = time_lib.time()
                    return latest_close
            
            self.logger.error(f"Failed to get NIFTY spot price from API. Response: {response}")
            raise ValueError("Failed to get NIFTY spot price from API")
            
        except Exception as e:
            self.logger.error(f"Error getting NIFTY spot price: {e}", exc_info=True)
            raise
    @rate_limited(limiter=lambda self: self.rate_limiter, key="get_recent_nifty_data")
    def get_recent_nifty_data(self, interval_minutes: int = 1, days: int = 1) -> pd.DataFrame:
        """
        Get recent NIFTY price data
        
        This function fetches recent NIFTY price data from the Breeze Connect API.
        
        Args:
            interval_minutes: Interval between data points in minutes (1, 5, 15, 30, 60, 1440)
            days: Number of days of data to fetch
            
        Returns:
            DataFrame with NIFTY price data (open, high, low, close, volume)
            
        Raises:
            ValueError: If the API call fails or returns invalid data
        """
        # Generate a unique cache key based on the interval and days
        cache_key = f"nifty_data_{interval_minutes}_{days}"
        
        # Clean expired cache entries
        self._clean_cache()
        
        # Check if data is present in cache and return it if available
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        try:
            # Check if Breeze Connect API is available and instantiated
            if not self.breeze_api and BREEZE_CONNECT_AVAILABLE:
                api_key = os.getenv('ICICI_API_KEY')
                api_secret = os.getenv('ICICI_API_SECRET')
                session_token = os.getenv('ICICI_SESSION_TOKEN')
                
                if api_key and api_secret and session_token:
                    self.breeze_api = BreezeConnect(api_key=api_key)
                    self.breeze_api.generate_session(api_secret=api_secret, session_token=session_token)
            
            # Calculate the date range for the historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates as strings for API request
            start_str = start_date.strftime("%Y-%m-%dT%H:%M:%S")
            end_str = end_date.strftime("%Y-%m-%dT%H:%M:%S")
            
            # Map interval_minutes to Breeze Connect interval format
            interval_map = {
                1: "1minute",
                5: "5minute",
                15: "15minute",
                30: "30minute",
                60: "1hour",
                1440: "1day"
            }
            
            interval = interval_map.get(interval_minutes, "1minute")
            
            # Fetch historical data from Breeze Connect API
            response = self.breeze_api.get_historical_data_v2(
                interval=interval,
                from_date=start_str,
                to_date=end_str,
                stock_code="NIFTY",
                exchange_code="NSE",
                product_type="cash"
            )
            
            # Log raw response for debugging
            self.logger.debug(f"Historical data API response: {response}")
            
            # Check if response contains data
            if self._validate_api_response(response):
                # For Breeze Connect, response is a list of dictionaries
                if isinstance(response, list) and len(response) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(response)
                    
                    # Convert datetime string to datetime index
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        df.set_index('datetime', inplace=True)
                    
                    # Ensure required columns exist
                    required_columns = ['open', 'high', 'low', 'close', 'volume']
                    for col in required_columns:
                        if col not in df.columns:
                            df[col] = 0.0
                    
                    # Convert numeric columns
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Sort by date (oldest first)
                    df = df.sort_index()
                    
                    # Cache the DataFrame result
                    self.cache[cache_key] = df
                    self.cache_timestamps[cache_key] = time_lib.time()
                    
                    return df
                else:
                    self.logger.error("No data returned in API response")
            else:
                self.logger.error(f"Invalid API response: {response}")
            
            # If we get here, the API call failed or returned no data
            self.logger.error("Failed to get NIFTY historical data from API")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            
        except Exception as e:
            self.logger.error(f"Error getting NIFTY historical data: {e}", exc_info=True)
            # Return an empty DataFrame in case of an error
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            
    @rate_limited(limiter=lambda self: self.rate_limiter, key="get_option_chain")
    def get_option_chain(self, expiry_date: date) -> Optional[pd.DataFrame]:
        """
        Get the NIFTY option chain for the given expiry date
        
        This function fetches the option chain data from the Breeze Connect API and returns
        it as a pandas DataFrame with the following columns:
        - strike_price: The strike price of the option
        - call_ltp: The last traded price of the call option
        - put_ltp: The last traded price of the put option
        - call_oi: The open interest of the call option
        - put_oi: The open interest of the put option
        - call_volume: The trading volume of the call option
        - put_volume: The trading volume of the put option
        - call_iv: The implied volatility of the call option
        - put_iv: The implied volatility of the put option
        
        Args:
            expiry_date: The expiry date of the options chain to fetch
            
        Returns:
            A pandas DataFrame containing the option chain data, or None if the data
            could not be fetched
        """
        try:
            # Generate a cache key based on the expiry date
            cache_key = f"option_chain_{expiry_date}"
            
            # Clean up expired cache entries
            self._clean_cache()
            
            # Return cached data if available
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Check if Breeze Connect API is available and instantiated
            if not self.breeze_api and BREEZE_CONNECT_AVAILABLE:
                api_key = os.getenv('ICICI_API_KEY')
                api_secret = os.getenv('ICICI_API_SECRET')
                session_token = os.getenv('ICICI_SESSION_TOKEN')
                
                if api_key and api_secret and session_token:
                    self.breeze_api = BreezeConnect(api_key=api_key)
                    self.breeze_api.generate_session(api_secret=api_secret, session_token=session_token)
            
            if not self.breeze_api:
                self.logger.error("Breeze Connect API not initialized")
                return None
            
            # Format the expiry date as a string in the required format (DD-MM-YYYY)
            expiry_str = expiry_date.strftime("%d-%m-%Y")
            
            # Get the option chain data from the Breeze Connect API
            response = self.breeze_api.get_option_chain_quotes(
                stock_code="NIFTY",
                exchange_code="NFO",
                product_type="options",
                expiry_date=expiry_str,
                right="both",
                strike_price=""
            )
            
            # Log raw response for debugging
            self.logger.debug(f"Option chain API response: {response}")
            
            # Check if the response is valid
            if not self._validate_api_response(response):
                self.logger.error(f"Invalid option chain response: {response}")
                return None
                
            # Extract the option chain data
            chain_data = response.get('Success', [])
            
            if not chain_data:
                self.logger.error("No option chain data in response")
                return None
            
            # Create a dictionary to store the option chain data
            option_chain = {}
            
            # Process each option in the chain
            for option in chain_data:
                try:
                    strike = float(option.get('strike_price', 0))
                    if strike == 0:
                        continue
                        
                    # Initialize the strike price entry if it doesn't exist
                    if strike not in option_chain:
                        option_chain[strike] = {
                            'strike_price': strike,
                            'call_ltp': None,
                            'put_ltp': None,
                            'call_oi': None,
                            'put_oi': None,
                            'call_volume': None,
                            'put_volume': None,
                            'call_iv': None,
                            'put_iv': None
                        }
                    
                    # Get the option type (CE or PE)
                    option_type = option.get('right', '').upper()
                    
                    # Update the call or put data based on the option type
                    if option_type == 'CALL':
                        option_chain[strike]['call_ltp'] = float(option.get('ltp', 0))
                        option_chain[strike]['call_oi'] = int(option.get('open_interest', 0))
                        option_chain[strike]['call_volume'] = int(option.get('total_traded_volume', 0))
                        option_chain[strike]['call_iv'] = float(option.get('implied_volatility', 0))
                    elif option_type == 'PUT':
                        option_chain[strike]['put_ltp'] = float(option.get('ltp', 0))
                        option_chain[strike]['put_oi'] = int(option.get('open_interest', 0))
                        option_chain[strike]['put_volume'] = int(option.get('total_traded_volume', 0))
                        option_chain[strike]['put_iv'] = float(option.get('implied_volatility', 0))
                except Exception as e:
                    self.logger.warning(f"Error processing option data: {e}")
                    continue
            
            # Convert the dictionary to a DataFrame
            if option_chain:
                # Convert to list of dicts and sort by strike price
                chain_list = sorted(option_chain.values(), key=lambda x: x['strike_price'])
                df = pd.DataFrame(chain_list)
                
                # Cache the result
                self.cache[cache_key] = df
                self.cache_timestamps[cache_key] = time_lib.time()
                
                return df
            else:
                self.logger.error("No valid option chain data found in response")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting option chain: {e}", exc_info=True)
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
        
        This method retrieves detailed information for a specific options contract
        using the Breeze Connect API. It handles both call and put options and includes
        comprehensive error handling and logging.
        
        Args:
            strike: Strike price of the option (will be rounded to nearest 50 for NIFTY)
            option_type: 'CE' for call options, 'PE' for put options
            expiry_date: Expiry date of the option contract
            
        Returns:
            Dictionary containing the option details including:
            - symbol: Option symbol
            - strike: Strike price
            - type: Option type ('CE' or 'PE')
            - ltp: Last traded price
            - volume: Trading volume
            - oi: Open interest
            - iv: Implied volatility
            - bid_price: Best bid price
            - ask_price: Best ask price
            - expiry_date: Expiry date in 'YYYY-MM-DD' format
            
        Raises:
            ValueError: If the option is not found or if there's an error in the API response
        """
        try:
            # Get option chain for the expiry date
            option_chain = self.get_option_chain(expiry_date)
            
            if option_chain is None or option_chain.empty:
                error_msg = "Failed to fetch option chain or empty chain returned"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Round strike to nearest 50 (NIFTY standard)
            strike = round(strike / 50) * 50
            
            # Map option type to Breeze Connect format
            option_type = option_type.upper()
            if option_type not in ['CE', 'PE']:
                error_msg = f"Invalid option type: {option_type}. Must be 'CE' or 'PE'"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Find the specific strike in the option chain
            strike_row = option_chain[option_chain['strike_price'] == strike]
            
            if strike_row.empty:
                # If exact strike not found, find the nearest strike
                self.logger.warning(f"Exact strike {strike} not found, finding nearest strike")
                option_chain['strike_diff'] = abs(option_chain['strike_price'] - strike)
                strike_row = option_chain.nsmallest(1, 'strike_diff')
                strike = int(strike_row['strike_price'].iloc[0])
                self.logger.info(f"Using nearest strike: {strike}")
            
            # Extract the relevant data based on option type
            option_data = {
                'symbol': self.get_option_symbol(strike, option_type, expiry_date),
                'strike': float(strike),
                'type': option_type,
                'expiry_date': expiry_date.strftime("%Y-%m-%d"),
                'underlying': 'NIFTY',
                'exchange': 'NFO'
            }
            
            # Add call or put specific data
            if option_type == 'CE':
                option_data.update({
                    'ltp': float(strike_row['call_ltp'].iloc[0]) if pd.notna(strike_row['call_ltp'].iloc[0]) else 0.0,
                    'volume': int(strike_row['call_volume'].iloc[0]) if pd.notna(strike_row['call_volume'].iloc[0]) else 0,
                    'oi': int(strike_row['call_oi'].iloc[0]) if pd.notna(strike_row['call_oi'].iloc[0]) else 0,
                    'iv': float(strike_row['call_iv'].iloc[0]) if pd.notna(strike_row['call_iv'].iloc[0]) else 0.0,
                    'bid_price': float(strike_row['call_ltp'].iloc[0]) if pd.notna(strike_row['call_ltp'].iloc[0]) else 0.0,
                    'ask_price': float(strike_row['call_ltp'].iloc[0]) if pd.notna(strike_row['call_ltp'].iloc[0]) else 0.0
                })
            else:  # PE
                option_data.update({
                    'ltp': float(strike_row['put_ltp'].iloc[0]) if pd.notna(strike_row['put_ltp'].iloc[0]) else 0.0,
                    'volume': int(strike_row['put_volume'].iloc[0]) if pd.notna(strike_row['put_volume'].iloc[0]) else 0,
                    'oi': int(strike_row['put_oi'].iloc[0]) if pd.notna(strike_row['put_oi'].iloc[0]) else 0,
                    'iv': float(strike_row['put_iv'].iloc[0]) if pd.notna(strike_row['put_iv'].iloc[0]) else 0.0,
                    'bid_price': float(strike_row['put_ltp'].iloc[0]) if pd.notna(strike_row['put_ltp'].iloc[0]) else 0.0,
                    'ask_price': float(strike_row['put_ltp'].iloc[0]) if pd.notna(strike_row['put_ltp'].iloc[0]) else 0.0
                })
            
            self.logger.debug(f"Retrieved option details: {option_data}")
            return option_data
            
        except Exception as e:
            error_msg = f"Error getting option details for {strike} {option_type}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e
    
    def get_option_quote(self, strike: int, option_type: str, expiry_date: date) -> float:
        """
        Get the current price of a specific option
        
        This method provides a lightweight way to get just the last traded price (LTP)
        of an option contract using the Breeze Connect API. It's more efficient than
        get_option_details when you only need the price.
        
        Args:
            strike: Strike price of the option (will be rounded to nearest 50 for NIFTY)
            option_type: 'CE' for call options, 'PE' for put options
            expiry_date: Expiry date of the option contract
            
        Returns:
            float: The last traded price of the option
            
        Raises:
            ValueError: If the option is not found or if there's an error in the API response
        """
        try:
            # Get the full option details
            option_details = self.get_option_details(strike, option_type, expiry_date)
            
            # Extract and return the last traded price
            ltp = option_details.get('ltp', 0.0)
            
            if ltp is None or ltp <= 0:
                self.logger.warning(f"Invalid LTP ({ltp}) for {strike} {option_type}")
                return 0.0
                
            return float(ltp)
            
        except Exception as e:
            error_msg = f"Error getting option quote for {strike} {option_type}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise ValueError(error_msg) from e
        return option_details.get('last_price', 0.0)

    def get_market_data(self) -> Tuple[pd.DataFrame, float]:
        """
        Fetch market data including OHLC candles and current spot price
        
        This method retrieves the latest market data including historical OHLC candles
        and the current spot price for NIFTY. It uses the Breeze Connect API to fetch
        this data and handles errors gracefully.
        
        Args:
            None
        
        Returns:
            Tuple[pd.DataFrame, float]: A tuple containing:
                - DataFrame with OHLCV (Open, High, Low, Close, Volume) data
                - Current spot price of NIFTY
                
        Raises:
            Exception: If there's an error fetching the market data
        """
        try:
            # Get configuration values with defaults
            interval_minutes = int(self.config.get('data', 'ohlc_timeframe', fallback='1'))
            days = int(self.config.get('data', 'max_history_days', fallback='1'))
            
            self.logger.debug(f"Fetching market data - Interval: {interval_minutes}min, Days: {days}")
            
            # Get historical OHLC data
            df = self.get_recent_nifty_data(interval_minutes, days)
            
            if df is None or df.empty:
                error_msg = "Failed to fetch OHLC data or empty data returned"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Get current spot price
            spot_price = self.get_nifty_spot()
            
            if spot_price <= 0:
                error_msg = "Invalid spot price received"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Log summary of the data
            self.logger.debug(f"Fetched {len(df)} OHLC records, latest at {df.index[-1]}")
            self.logger.debug(f"Current NIFTY spot price: {spot_price}")
            
            return df, spot_price
            
        except Exception as e:
            error_msg = f"Error getting market data: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise Exception(error_msg) from e

