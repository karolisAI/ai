import logging
import time
from datetime import datetime
from functools import wraps
import os

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fitness_bot.log'),
        logging.StreamHandler()
    ]
)

class RateLimiter:
    def __init__(self, max_calls, time_window):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # remove calls outside the time window
            self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            
            if len(self.calls) >= self.max_calls:
                logging.warning(f"Rate limit exceeded for {func.__name__}")
                raise Exception("Rate limit exceeded. Please try again later.")
            
            self.calls.append(now)
            return func(*args, **kwargs)
        return wrapper

def log_query(user_input, response, source=None):
    """Log user queries and responses"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'user_input': user_input,
        'response': response,
        'source': source
    }
    logging.info(f"Query processed: {log_entry}")

def validate_input(user_input):
    """Validate user input for security and content"""
    if not user_input or len(user_input.strip()) == 0:
        raise ValueError("Input cannot be empty")
    
    if len(user_input) > 1000:
        raise ValueError("Input is too long. Please keep it under 1000 characters")
    
    return user_input.strip()

# initialize rate limiter
rate_limiter = RateLimiter(max_calls=10, time_window=60)  # 10 calls per minute 