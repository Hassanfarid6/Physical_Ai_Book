"""
Performance monitoring and metrics for the book embeddings ingestion pipeline.
"""
import time
import sys
import os
from typing import Callable, Any
from functools import wraps

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging import setup_logging

logger = setup_logging()


class MetricsCollector:
    """Collects and reports performance metrics for the ingestion pipeline."""

    def __init__(self):
        self.metrics = {}

    def record_duration(self, operation: str, duration: float):
        """Record the duration of an operation."""
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)

    def get_average_duration(self, operation: str) -> float:
        """Get the average duration for an operation."""
        if operation not in self.metrics or not self.metrics[operation]:
            return 0.0
        return sum(self.metrics[operation]) / len(self.metrics[operation])

    def get_total_operations(self, operation: str) -> int:
        """Get the total number of operations performed."""
        return len(self.metrics.get(operation, []))

    def log_metrics(self):
        """Log all collected metrics."""
        for operation, durations in self.metrics.items():
            avg_duration = self.get_average_duration(operation)
            total_ops = self.get_total_operations(operation)
            logger.info(f"Metrics for {operation}: {total_ops} operations, "
                       f"avg duration: {avg_duration:.4f}s")


# Global metrics collector
metrics_collector = MetricsCollector()


def measure_time(operation_name: str):
    """
    Decorator to measure the execution time of a function.

    Args:
        operation_name: Name of the operation being measured
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                metrics_collector.record_duration(operation_name, duration)
                logger.debug(f"{operation_name} took {duration:.4f}s")
        return wrapper
    return decorator


def get_performance_report() -> dict:
    """
    Get a performance report with collected metrics.

    Returns:
        Dictionary containing performance metrics
    """
    report = {}
    for operation in metrics_collector.metrics:
        report[operation] = {
            'total_operations': metrics_collector.get_total_operations(operation),
            'average_duration': metrics_collector.get_average_duration(operation),
            'total_duration': sum(metrics_collector.metrics[operation])
        }
    return report