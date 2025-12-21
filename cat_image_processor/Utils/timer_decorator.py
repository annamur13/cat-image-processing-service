import time
from functools import wraps


def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        class_name = args[0].__class__.__name__ if args else ""
        print(f"Starting {class_name}.{func.__name__}...")

        result = func(*args, **kwargs)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Finished {class_name}.{func.__name__} in {execution_time:.4f} seconds")

        return result

    return wrapper