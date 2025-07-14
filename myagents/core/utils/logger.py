import sys
import os
import uuid
from datetime import datetime

from loguru import logger

from myagents.core.interface import Logger

# 全局变量跟踪是否已经初始化
_logger_initialized = False

def get_colored_format(record):
    """根据日志等级返回不同颜色的格式"""
    time_part = "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
    level_part = "<level>{level:<8}</level>"
    
    # 根据日志等级选择消息部分的颜色
    level_name = record["level"].name
    if level_name == "DEBUG":
        message_part = "<cyan>{name}:{function}:{line} - {message}</cyan>"
    elif level_name == "INFO":
        message_part = "<white>{name}:{function}:{line} - {message}</white>"
    elif level_name == "WARNING":
        message_part = "<yellow>{name}:{function}:{line} - {message}</yellow>"
    elif level_name in ["ERROR", "CRITICAL"]:
        message_part = "<red>{name}:{function}:{line} - {message}</red>"
    else:
        message_part = "<cyan>{name}:{function}:{line} - {message}</cyan>"
    
    # 添加换行符确保每条日志都换行
    return f"{time_part} | {level_part} | {message_part}\n"


def init_logger(level: str = "INFO", sink: str = "stdout", task_name: str = None, **kwargs) -> Logger:
    global _logger_initialized
    
    # 如果已经初始化过，先移除所有现有的处理器
    if _logger_initialized:
        logger.remove()
    
    # 移除所有默认的处理器，确保完全替换
    logger.remove()
    
    if sink == "stdout":
        # Add the new logger with dynamic color based on level
        logger.add(
            sys.stdout, 
            format=get_colored_format, 
            level=level, 
            colorize=True,
            # 添加自动换行设置
            backtrace=True,
            diagnose=True
        )
    else:
        # Check if sink is a directory and create it if needed
        if not os.path.exists(sink):
            os.makedirs(sink)
        
        # Generate task name if not provided
        if task_name is None:
            task_name = str(uuid.uuid4())[:8]  # 使用UUID前8位作为任务名
        
        # Construct log file path with task name and timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(sink, f"{task_name}_{timestamp}.log")
        
        # Modify kwargs
        kwargs["level"] = kwargs.get("level", level)
        kwargs["colorize"] = kwargs.get("colorize", True)
        kwargs["format"] = kwargs.get("format", get_colored_format)
        kwargs["backtrace"] = kwargs.get("backtrace", True)
        kwargs["diagnose"] = kwargs.get("diagnose", True)
        
        # Log to designated file with daily rotation
        logger.add(log_file, **kwargs)

    # 标记为已初始化
    _logger_initialized = True
    
    return logger


def reset_logger():
    """重置 logger 到初始状态，移除所有处理器"""
    global _logger_initialized
    logger.remove()
    _logger_initialized = False
