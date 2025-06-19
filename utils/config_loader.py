import os
import configparser
from typing import Optional, Dict, Any, List


class ConfigLoader:
    """
    Configuration loader utility for loading and validating INI config files
    """
    
    def __init__(self, config_file: str):
        """
        Initialize the config loader with a config file path
        
        Args:
            config_file: Path to the configuration file
        """
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        self.config.read(config_file)
    
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
