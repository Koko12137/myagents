from typing import Any, Optional

from pydantic import BaseModel


class BaseContext(BaseModel):
    """BaseContext is the base class for all the contexts.
    
    Attributes:
        prev (BaseContext):
            The previous task or environment.
        next (BaseContext):
            The next task or environment.
        key_values (dict[str, Any]):
            The key-value pairs of the context.
    """
    prev: Optional['BaseContext'] = None
    next: Optional['BaseContext'] = None
    key_values: dict[str, Any]
    
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
