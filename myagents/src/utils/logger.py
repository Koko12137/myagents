import sys
import os
from datetime import datetime

from loguru import logger

from myagents.src.interface import Logger


logger_prompt = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level:<8}</level> | "
    "<cyan>{name}:{function}:{line} - {message}</cyan>"
)


def init_logger(level: str = "INFO", sink: str = "stdout", **kwargs) -> Logger:
    # Remove the default logger
    logger.remove()
    
    if sink == "stdout":
        # Add the new logger
        logger.add(sys.stdout, format=logger_prompt, level=level, colorize=True)
    else:
        # Check if sink is a directory and create it if needed
        if not os.path.exists(sink):
            os.makedirs(sink)
        
        # Construct log file path with timestamp
        log_file = os.path.join(sink, f"log_{datetime.now().strftime('%Y%m%d')}.log")
        
        # Modify kwargs
        kwargs["rotation"] = kwargs.get("rotation", "00:00") # Rotate at midnight
        kwargs["retention"] = kwargs.get("retention", "30 days")
        kwargs["compression"] = kwargs.get("compression", "zip")
        kwargs["level"] = kwargs.get("level", level)
        kwargs["colorize"] = kwargs.get("colorize", True)
        kwargs["format"] = kwargs.get("format", logger_prompt)
        
        # Log to designated file with daily rotation
        logger.add(log_file, **kwargs)

    return logger
