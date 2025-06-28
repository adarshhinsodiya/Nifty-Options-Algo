import os
import configparser
import logging
from typing import Optional, Dict, Any, List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Configuration loader utility for loading and validating INI config files
    """
    
    DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "default_config.ini")
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the config loader with a config file path and environment variables
        
        Args:
            config_file: Path to the configuration file (optional)
        """
        self.config_file = config_file or self.DEFAULT_CONFIG_PATH
        self.config = configparser.ConfigParser()
        
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
            
        self.config.read(self.config_file)
        self._validate_breeze_connect_config()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a section of the configuration as a dictionary
        
        Args:
            section: Name of the section to retrieve
            
        Returns:
            Dictionary containing the section's key-value pairs
        """
        if not self.config.has_section(section):
            raise ValueError(f"Section '{section}' not found in config file")
            
        return dict(self.config[section])
    
    def validate_required_fields(self, section: str, required_fields: List[str]) -> bool:
        """
        Validate that all required fields exist in a section
        
        Args:
            section: Name of the section to validate
            required_fields: List of field names that must exist
            
        Returns:
            True if all required fields exist, False otherwise
        """
        if not self.config.has_section(section):
            return False
            
        section_dict = dict(self.config[section])
        return all(field in section_dict for field in required_fields)
    
    def get(self, section: str, option: str, fallback: Any = None) -> Any:
        """
        Get a specific option from a section with fallback
        
        Args:
            section: Name of the section
            option: Name of the option
            fallback: Default value if option doesn't exist
            
        Returns:
            Value of the option or fallback if not found
        """
        return self.config.get(section, option, fallback=fallback)
    
    def getint(self, section: str, option: str, fallback: Optional[int] = None) -> int:
        """
        Get an integer option from a section with fallback
        
        Args:
            section: Name of the section
            option: Name of the option
            fallback: Default value if option doesn't exist
            
        Returns:
            Integer value of the option or fallback if not found
        """
        return self.config.getint(section, option, fallback=fallback)
    
    def getfloat(self, section: str, option: str, fallback: Optional[float] = None) -> float:
        """
        Get a float option from a section with fallback
        
        Args:
            section: Name of the section
            option: Name of the option
            fallback: Default value if option doesn't exist
            
        Returns:
            Float value of the option or fallback if not found
        """
        return self.config.getfloat(section, option, fallback=fallback)
    
    def getboolean(self, section: str, option: str, fallback: Optional[bool] = None) -> bool:
        """
        Get a boolean option from a section with fallback
        
        Args:
            section: Name of the section
            option: Name of the option
            fallback: Default value if option doesn't exist
            
        Returns:
            Boolean value of the option or fallback if not found
        """
        return self.config.getboolean(section, option, fallback=fallback)
        
    def _validate_breeze_connect_config(self) -> None:
        """
        Validate Breeze Connect API configuration from environment variables.
        
        Raises:
            ValueError: If required Breeze Connect environment variables are missing
        """
        if self.config.getboolean('mode', 'live', fallback=False):
            required_vars = ['ICICI_API_KEY', 'ICICI_API_SECRET', 'ICICI_SESSION_TOKEN']
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                error_msg = f"Missing required environment variables for Breeze Connect: {', '.join(missing_vars)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info("Breeze Connect API configuration validated successfully")
    
    def get_breeze_connect_config(self) -> Dict[str, str]:
        """
        Get Breeze Connect API configuration from environment variables.
        
        Returns:
            Dict containing Breeze Connect configuration
        """
        return {
            'api_key': os.getenv('ICICI_API_KEY', ''),
            'api_secret': os.getenv('ICICI_API_SECRET', ''),
            'session_token': os.getenv('ICICI_SESSION_TOKEN', ''),
            'user_id': os.getenv('ICICI_USER_ID', '')  # Optional
        }
