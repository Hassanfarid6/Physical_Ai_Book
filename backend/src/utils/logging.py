import logging
import sys
from typing import Optional

def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level as a string (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR')
        
    Returns:
        Configured logger instance
    """
    # Determine logging level
    log_level = getattr(logging, level.upper()) if level else logging.INFO
    
    # Create logger
    logger = logging.getLogger('book_embeddings_ingestion')
    logger.setLevel(log_level)
    
    # Avoid adding multiple handlers if logger already configured
    if logger.handlers:
        return logger
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger