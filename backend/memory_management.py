"""
Memory management utilities for the ingestion pipeline.
Provides functions to manage memory usage during processing of large documents.
"""
import psutil
import gc
import logging
from typing import Any, List
import sys


def get_memory_usage() -> float:
    """
    Get the current memory usage of the process as a percentage.
    
    Returns:
        Memory usage as a percentage
    """
    process = psutil.Process()
    return process.memory_percent()


def get_memory_info() -> dict:
    """
    Get detailed memory information about the current process.
    
    Returns:
        Dictionary with memory information
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss,  # Resident Set Size (physical memory being used)
        'vms': memory_info.vms,  # Virtual Memory Size (total virtual memory used)
        'percent': process.memory_percent()
    }


def is_memory_usage_high(threshold: float = 80.0) -> bool:
    """
    Check if memory usage is above the specified threshold.
    
    Args:
        threshold: Memory usage threshold percentage (default 80%)
        
    Returns:
        True if memory usage is above threshold, False otherwise
    """
    return get_memory_usage() > threshold


def force_garbage_collection():
    """
    Force garbage collection to free up memory.
    """
    collected = gc.collect()
    logging.info(f"Garbage collection performed, collected {collected} objects")


def process_in_batches(items: List[Any], batch_size: int, process_func, *args, **kwargs):
    """
    Process a large list of items in batches to manage memory usage.
    
    Args:
        items: List of items to process
        batch_size: Number of items to process in each batch
        process_func: Function to call for processing each batch
        *args: Additional arguments to pass to process_func
        **kwargs: Additional keyword arguments to pass to process_func
        
    Returns:
        List of results from processing all batches
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Check memory usage before processing the batch
        memory_info = get_memory_info()
        logging.debug(f"Memory usage before batch {i//batch_size + 1}: {memory_info['percent']:.2f}%")
        
        # Process the batch
        batch_result = process_func(batch, *args, **kwargs)
        results.extend(batch_result)
        
        # Force garbage collection after each batch to free memory
        force_garbage_collection()
        
        # Check memory usage after processing the batch
        memory_info = get_memory_info()
        logging.debug(f"Memory usage after batch {i//batch_size + 1}: {memory_info['percent']:.2f}%")
    
    return results


def monitor_memory_usage(threshold: float = 80.0) -> bool:
    """
    Monitor memory usage and log warnings if it exceeds the threshold.
    
    Args:
        threshold: Memory usage threshold percentage (default 80%)
        
    Returns:
        True if memory usage is within acceptable limits, False otherwise
    """
    memory_percent = get_memory_usage()
    
    if memory_percent > threshold:
        logging.warning(f"High memory usage detected: {memory_percent:.2f}%")
        return False
    else:
        return True


def get_largest_objects(limit: int = 10) -> List[tuple]:
    """
    Get the largest objects in memory for debugging purposes.
    
    Args:
        limit: Number of largest objects to return (default 10)
        
    Returns:
        List of tuples containing (object_type, size_in_bytes)
    """
    objects = gc.get_objects()
    object_sizes = []
    
    for obj in objects:
        try:
            size = sys.getsizeof(obj)
            obj_type = type(obj).__name__
            object_sizes.append((obj_type, size))
        except:
            # Skip objects that we can't get the size of
            continue
    
    # Sort by size in descending order and return the top 'limit' items
    object_sizes.sort(key=lambda x: x[1], reverse=True)
    return object_sizes[:limit]