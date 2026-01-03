"""
Logging and progress tracking utilities.
Provides functions for logging and tracking progress of the ingestion pipeline.
"""
import logging
from datetime import datetime
from typing import Optional
import sys
import os


def setup_logging(log_level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional file path to write logs to
    """
    # Create a custom format for logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),  # Log to console
        ]
    )
    
    # Add file handler if log_file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


class ProgressTracker:
    """
    A utility class to track progress of long-running operations.
    """
    
    def __init__(self, total_items: int, operation_name: str = "Operation"):
        """
        Initialize the progress tracker.
        
        Args:
            total_items: Total number of items to process
            operation_name: Name of the operation for logging
        """
        self.total_items = total_items
        self.operation_name = operation_name
        self.completed_items = 0
        self.start_time = datetime.now()
        
    def update(self, increment: int = 1, message: str = "") -> None:
        """
        Update the progress with the number of completed items.
        
        Args:
            increment: Number of items completed since last update (default: 1)
            message: Optional message to log with the progress
        """
        self.completed_items += increment
        elapsed_time = datetime.now() - self.start_time
        
        # Calculate percentage
        percentage = (self.completed_items / self.total_items) * 100 if self.total_items > 0 else 0
        
        # Log progress
        progress_msg = (
            f"{self.operation_name}: {self.completed_items}/{self.total_items} "
            f"({percentage:.1f}%) - Elapsed: {elapsed_time}"
        )
        
        if message:
            progress_msg += f" - {message}"
            
        logging.info(progress_msg)
        
    def complete(self) -> None:
        """
        Mark the operation as complete and log final stats.
        """
        elapsed_time = datetime.now() - self.start_time
        logging.info(
            f"{self.operation_name} completed: {self.completed_items}/{self.total_items} "
            f"items processed in {elapsed_time}"
        )


def log_operation_start(operation_name: str) -> datetime:
    """
    Log the start of an operation and return the start time.
    
    Args:
        operation_name: Name of the operation to log
        
    Returns:
        Start time of the operation
    """
    start_time = datetime.now()
    logging.info(f"Starting {operation_name} at {start_time}")
    return start_time


def log_operation_end(operation_name: str, start_time: datetime) -> None:
    """
    Log the end of an operation with duration.
    
    Args:
        operation_name: Name of the operation to log
        start_time: Start time of the operation (from log_operation_start)
    """
    end_time = datetime.now()
    duration = end_time - start_time
    logging.info(f"Completed {operation_name} at {end_time} (Duration: {duration})")


def log_error_context(context: str, error: Exception) -> None:
    """
    Log error with context information.
    
    Args:
        context: Context information about where the error occurred
        error: The exception that occurred
    """
    logging.error(f"Error in {context}: {str(error)}", exc_info=True)