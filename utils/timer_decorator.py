"""
Timer decorator for measuring function execution time
"""

import time
import functools

def timer_decorator(func):
    """Decorator to measure function execution time
    
    Args:
        func: The function to be decorated
        
    Returns:
        wrapper: The wrapped function that includes timing functionality
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' took {execution_time:.2f} seconds to execute")
        return result
    return wrapper
