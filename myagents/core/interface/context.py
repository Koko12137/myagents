from abc import abstractmethod
from typing import Protocol, runtime_checkable, Any, Optional


@runtime_checkable
class Context(Protocol):
    """Context records the runtime information for global context. It is used to pass the information between 
    tool calling and the workflow. It can also be a general variable container for the life cycle of the workflow.  
    One context contains key-value pairs that set by the workflow, and the context visibility can be controlled 
    by layer of the context. 
    
    Attributes:
        prev (Context):
            The previous context.
        next (Context):
            The next context.
        key_values (dict[str, Any]):
            The key values of the context.
    """
    prev: Optional['Context']
    next: Optional['Context']
    key_values: dict[str, Any]
    
    @abstractmethod
    def append(self, key: str, value: Any) -> None:
        """Append a key-value pair to the context.
        
        Args:
            key (str):
                The key of the value.
            value (Any):
                The value of the key.
        """
        pass
    
    @abstractmethod
    def update(self, key: str, value: Any) -> None:
        """Update the value of the key.
        
        Args:
            key (str):
                The key of the value.
            value (Any):
                The value of the key.
        """
        pass
    
    @abstractmethod
    def get(self, key: str) -> Any:
        """Get the value of the key.
        
        Args:
            key (str):
                The key of the value.

        Returns:
            Any:
                The value of the key.
        """
        pass
    
    @abstractmethod
    def pop(self, key: str) -> Any:
        """Pop the value of the key.
        
        Args:
            key (str):
                The key of the value.
                
        Returns:
            Any:
                The value of the key.
        """
        pass
    
    @abstractmethod
    def create_next(self, **kwargs) -> 'Context':
        """Create the next context.
        
        Args:
            **kwargs:
                The keyword arguments to create the next context.
                
        Returns:
            Context:
                The next context.
        """
        pass
    
    @abstractmethod
    def done(self) -> 'Context':
        """Done the context and return the previous context.
        
        Returns:
            Context:
                The previous context.
        """
        pass