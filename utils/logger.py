import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional, Union


def setup_logger(name: str, config: Union[str, dict] = 'INFO', log_dir: str = 'logs') -> logging.Logger:
    """
    Setup comprehensive logging with proper formatting
    
    Args:
        name: Name of the logger
        config: Either a string log level or a config dict with logging settings
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Handle both string and config dict input
    if isinstance(config, str):
        log_level_str = config.upper()
    else:
        log_level_str = config.get('level', 'INFO').upper()
        
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if enabled
    if isinstance(config, dict) and config.get('log_to_file', False):
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Get file handler settings
        max_log_size = int(config.get('max_log_size_mb', 10)) * 1024 * 1024  # Convert to bytes
        backup_count = int(config.get('backup_count', 3))
        
        # Create file handler
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_log_size,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
