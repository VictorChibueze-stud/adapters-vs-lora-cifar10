"""
Sophisticated logging and error handling utilities.
"""
import logging
import sys
from typing import Optional

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Configure logging for the project.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        handlers=handlers
    )

class ExceptionHandler:
    """
    Context manager for robust exception handling and logging.
    """
    def __init__(self, context: str = "Execution"):
        self.context = context
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logging.error(f"Exception in {self.context}: {exc_val}", exc_info=True)
        return False  # Do not suppress exceptions
