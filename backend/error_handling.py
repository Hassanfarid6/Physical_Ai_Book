"""
Error handling and retry utilities.
Provides functions for handling errors gracefully and implementing retry logic.
"""
import time
import logging
from functools import wraps
from typing import Callable, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def create_session_with_retries(
    total_retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: tuple = (500, 502, 504)
) -> requests.Session:
    """
    Create a requests session with retry strategy.
    
    Args:
        total_retries: Total number of retries to attempt
        backoff_factor: Backoff factor for exponential backoff
        status_forcelist: Tuple of HTTP status codes to retry on
        
    Returns:
        A requests.Session object with retry strategy configured
    """
    session = requests.Session()
    
    # Define retry strategy
    retry_strategy = Retry(
        total=total_retries,
        status_forcelist=status_forcelist,
        backoff_factor=backoff_factor,
        # Don't raise exception on redirect, just retry
        raise_on_redirect=False,
        # Don't raise exception on status, just retry
        raise_on_status=False
    )
    
    # Mount adapter with retry strategy
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


def retry_on_exception(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry a function if it raises specific exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry on
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retries = 0
            current_delay = delay
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    if retries >= max_retries:
                        logging.error(f"Function {func.__name__} failed after {max_retries} retries: {str(e)}")
                        raise e
                    
                    logging.warning(
                        f"Function {func.__name__} failed (attempt {retries}/{max_retries}): {str(e)}. "
                        f"Retrying in {current_delay} seconds..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator


def handle_api_error(response: requests.Response, context: str = "") -> None:
    """
    Handle API errors by logging them appropriately.
    
    Args:
        response: The response object from the API call
        context: Context information for logging
    """
    if not response.ok:
        error_msg = f"API request failed ({response.status_code}): {response.text}"
        if context:
            error_msg = f"{context} - {error_msg}"
        logging.error(error_msg)


def validate_response(response: requests.Response, expected_status: int = 200) -> bool:
    """
    Validate that a response has the expected status code.
    
    Args:
        response: The response object to validate
        expected_status: The expected status code (default 200)
        
    Returns:
        True if status code matches expected, False otherwise
    """
    if response.status_code != expected_status:
        logging.warning(f"Expected status {expected_status}, got {response.status_code}")
        return False
    return True


def safe_execute(func: Callable, *args, error_message: str = "", default_return: Any = None) -> Any:
    """
    Safely execute a function, catching exceptions and returning a default value.
    
    Args:
        func: The function to execute
        *args: Arguments to pass to the function
        error_message: Error message to log if function fails
        default_return: Default value to return if function fails
        
    Returns:
        Result of the function or default value if it fails
    """
    try:
        return func(*args)
    except Exception as e:
        if error_message:
            logging.error(f"{error_message}: {str(e)}")
        else:
            logging.error(f"Function execution failed: {str(e)}")
        return default_return