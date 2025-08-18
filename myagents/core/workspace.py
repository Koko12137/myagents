from typing import Any

from myagents.core.interface import Workspace


class BaseWorkspace(Workspace):
    """工作空间基类，用于记录全局可访问的变量
    
    方法: 
        add(sub_space_id: str, key: str, value: Any) -> None:
            添加一个键值对到工作空间
        update(sub_space_id: str, key: str, value: Any) -> None:
            更新一个键值对
        get(sub_space_id: str, key: str, default: Any) -> Any:
            获取一个键值对
        pop(sub_space_id: str, key: str) -> Any:
            删除一个键值对
    """
    key_values: dict[str, dict[str, Any]]
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.key_values = {}
    
    def add(self, sub_space_id: str, key: str, value: Any) -> None:
        """Append the key-value pair to the workspace.
        
        Args:
            sub_space_id (str):
                The sub-space id of the workspace.
            key (str):
                The key of the value.
            value (Any):
                The value of the key.
                
        Raises:
            ValueError:
                The key already exists.
        """
        # Check if the sub-space id exists
        if sub_space_id not in self.key_values:
            self.key_values[sub_space_id] = {}
            
        # Check if the key already exists
        if key in self.key_values[sub_space_id]:
            raise ValueError(f"The key {key} already exists. Use `update` to update the value.")
        
        self.key_values[sub_space_id][key] = value

    def get(self, sub_space_id: str, key: str, default: Any = None) -> Any:
        """Get the value of the workspace.
        
        Args:
            key (str):
                The key of the value.
            default (Any, optional):
                The default value if the key is not found.
                
        Returns:
            Any:
                The value of the workspace.
        """
        return self.key_values.get(sub_space_id, {}).get(key, default)
    
    def update(self, sub_space_id: str, key: str, value: Any) -> None:
        """Update the value of the key.
        
        Args:
            key (str):
                The key of the value.
            value (Any):
                The value of the key.
        """
        # Check if the sub-space id exists
        if sub_space_id not in self.key_values:
            self.key_values[sub_space_id] = {}
        
        self.key_values[sub_space_id][key] = value
    
    def pop(self, sub_space_id: str, key: str) -> None:
        """Remove the key-value pair from the workspace.
        
        Args:
            key (str):
                The key of the value.
                
        Raises:
            KeyError:
                If the key is not found.
        """
        self.key_values[sub_space_id].pop(key)
