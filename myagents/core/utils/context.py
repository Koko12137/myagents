from typing import Any, Optional

from pydantic import BaseModel, Field


class BaseContext(BaseModel):
    """BaseContext contains the global variables for the workflow. 
    
    Attributes:
        prev (BaseContext):
            The previous task or environment.
        next (BaseContext):
            The next task or environment.
        key_values (dict[str, Any]):
            The key-value pairs of the context.
    """
    prev: Optional['BaseContext'] = Field(default=None)
    next: Optional['BaseContext'] = Field(default=None)
    key_values: dict[str, Any] = Field(default_factory=dict)
    
    def append(self, key: str, value: Any) -> None:
        """Append the key-value pair to the context.
        
        Args:
            key (str):
                The key of the value.
            value (Any):
                The value of the key.
                
        Raises:
            ValueError:
                The key already exists.
        """
        if key in self.key_values:
            raise ValueError(f"The key {key} already exists. Use `update` to update the value.")
        
        self.key_values[key] = value

    def get(self, key: str) -> Any:
        """Get the value of the context.
        
        Args:
            key (str):
                The key of the value.
                
        Returns:
            Any:
                The value of the context.
                
        Raises:
            KeyError:
                If the key is not found.
        """
        return self.key_values.get(key)
    
    def update(self, key: str, value: Any) -> None:
        """Update the value of the key.
        
        Args:
            key (str):
                The key of the value.
            value (Any):
                The value of the key.
        """
        self.key_values[key] = value
    
    def pop(self, key: str) -> None:
        """Remove the key-value pair from the context.
        
        Args:
            key (str):
                The key of the value.
                
        Raises:
            KeyError:
                If the key is not found.
        """
        self.key_values.pop(key)
    
    def create_next(self, **kwargs: dict) -> 'BaseContext':
        """Create the next tool call context.
        
        Args:
            **kwargs (dict):
                The keyword arguments to create the next tool call context.
                
        Returns:
            BaseContext:
                The next workflow context.
        """
        return BaseContext(
            prev=self,
            next=None,
            key_values=kwargs
        )
        
    def done(self) -> 'BaseContext':
        """Done the context and return the previous context.
        
        Returns:
            BaseContext:
                The previous tool call context.
        """
        return self.prev
