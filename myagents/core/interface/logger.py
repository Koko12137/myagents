from abc import abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class Logger(Protocol):
    """Logger is the protocol for the logger.
    """
    @abstractmethod
    def add(self, sink: str, format: str, level: str, colorize: bool, **kwargs) -> None:
        """Add a sink to the logger.
        
        Args:
            sink (str):
                The sink to add.
            format (str):
                The format of the sink.
            level (str):
                The level of the sink.
            colorize (bool):
                Whether to colorize the sink.
            **kwargs:
                The additional keyword arguments to pass to the add method.
        """
        pass
    
    @abstractmethod
    def enable(self, name: str) -> None:
        """Enable the logger. This is used to enable the logger for a specific name.
        
        Args:
            name (str):
                The name of the logger.
        """
        pass
    
    @abstractmethod
    def debug(self, message: str) -> None:
        """Debug the message. This is the lowest level of the logger.
        
        Args:
            message (str):
                The message to debug.
        """
        pass
    
    @abstractmethod
    def info(self, message: str) -> None:
        """Info the message. This is the second lowest level of the logger.  
        
        This is the default level of the logger.
        
        Args:
            message (str):
                The message to info.
        """
        pass
    
    @abstractmethod
    def warning(self, message: str) -> None:
        """Warning the message. This is the third lowest level of the logger.  
        
        Args:
            message (str):
                The message to warning.
        """
        pass
    
    @abstractmethod
    def error(self, message: str) -> None:
        """Error the message. This is the fourth lowest level of the logger.
        
        Args:
            message (str):
                The message to error.
        """
        pass
    
    @abstractmethod
    def critical(self, message: str) -> None:
        """Critical the message. This is the highest level of the logger.
        
        Args:
            message (str):
                The message to critical.
        """
        pass
