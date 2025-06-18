import logging
import time
from datetime import datetime
from functools import wraps
import streamlit as st
from collections import defaultdict

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
        self.session_calls = defaultdict(list)
        self.last_warning_time = defaultdict(float)
        self.warning_cooldown = 5  # seconds between warnings

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get session ID or create one if it doesn't exist
            if 'session_id' not in st.session_state:
                st.session_state.session_id = str(time.time())
            session_id = st.session_state.session_id
            
            now = time.time()
            # remove calls outside the time window for this session
            self.session_calls[session_id] = [
                call_time for call_time in self.session_calls[session_id] 
                if now - call_time < self.time_window
            ]
            
            if len(self.session_calls[session_id]) >= self.max_calls:
                # Only show warning if enough time has passed since last warning for this session
                if now - self.last_warning_time[session_id] > self.warning_cooldown:
                    remaining_time = int(self.time_window - (now - self.session_calls[session_id][0]))
                    warning_msg = f"Rate limit reached. Please wait {remaining_time} seconds before trying again."
                    logging.warning(f"Rate limit exceeded for {func.__name__} in session {session_id}")
                    st.warning(warning_msg)
                    self.last_warning_time[session_id] = now
                raise Exception("Rate limit exceeded. Please try again later.")
            
            self.session_calls[session_id].append(now)
            return func(*args, **kwargs)
        return wrapper

def log_query(user_input, response, source=None):
    """Log user queries and responses"""
    try:
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'response': response if isinstance(response, str) else str(response),
            'source': source
        }
        logging.info(f"Query processed: {log_entry}")
    except Exception as e:
        logging.error(f"Error logging query: {str(e)}")

def validate_input(user_input):
    """Validate user input for security and content"""
    if not user_input or len(user_input.strip()) == 0:
        raise ValueError("Input cannot be empty")
    
    if len(user_input) > 1000:
        raise ValueError("Input is too long. Please keep it under 1000 characters")
    
    return user_input.strip()

def validate_embedding_dimensions(embedding_dim, collection_dim):
    """Validate that embedding dimensions match collection dimensions"""
    if not isinstance(embedding_dim, int) or not isinstance(collection_dim, int):
        error_msg = f"Invalid dimension types: embedding_dim={type(embedding_dim)}, collection_dim={type(collection_dim)}"
        logging.error(error_msg)
        raise TypeError(error_msg)
    
    if embedding_dim <= 0 or collection_dim <= 0:
        error_msg = f"Invalid dimension values: embedding_dim={embedding_dim}, collection_dim={collection_dim}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    if embedding_dim != collection_dim:
        error_msg = f"Embedding dimension {embedding_dim} does not match collection dimensionality {collection_dim}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    logging.info(f"Embedding dimensions validated: {embedding_dim}")
    return True

def validate_embeddings(embeddings, expected_dim=None):
    """Validate embeddings format and dimensions"""
    if not embeddings:
        error_msg = "Empty embeddings list"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    if not all(isinstance(emb, list) for emb in embeddings):
        error_msg = "Invalid embedding format: all embeddings must be lists"
        logging.error(error_msg)
        raise TypeError(error_msg)
    
    if not all(isinstance(val, (int, float)) for emb in embeddings for val in emb):
        error_msg = "Invalid embedding values: all values must be numbers"
        logging.error(error_msg)
        raise TypeError(error_msg)
    
    if expected_dim is not None:
        if not all(len(emb) == expected_dim for emb in embeddings):
            error_msg = f"Inconsistent embedding dimensions: expected {expected_dim}"
            logging.error(error_msg)
            raise ValueError(error_msg)
    
    logging.info(f"Validated {len(embeddings)} embeddings")
    return True

def log_embedding_operation(operation_type, batch_size, total_items, success=True, error=None):
    """Log embedding operation details"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'operation': operation_type,
        'batch_size': batch_size,
        'total_items': total_items,
        'success': success,
        'error': str(error) if error else None
    }
    if success:
        logging.info(f"Embedding operation completed: {log_entry}")
    else:
        logging.error(f"Embedding operation failed: {log_entry}")

# initialize rate limiter with more lenient settings
rate_limiter = RateLimiter(max_calls=50, time_window=60)  # 50 calls per minute per session 