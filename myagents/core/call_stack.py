from typing import Any

from myagents.core.interface import CallStack


class BaseCallStackItem:
    """BaseCallStackItem 是调用栈中的一个元素。
    """
    next: 'BaseCallStackItem'
    prev: 'BaseCallStackItem'
    key_values: dict[str, Any]
    
    def __init__(
        self, 
        next: 'BaseCallStackItem' = None, 
        prev: 'BaseCallStackItem' = None, 
        key_values: dict[str, Any] = None,
    ) -> None:
        # Initialize the next and prev
        self.next = next
        self.prev = prev
        # Initialize the key values
        self.key_values = key_values


class BaseCallStack(CallStack):
    """CallStack 是用于管理调用栈的混合类。
    """
    
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # 创建一个调用栈的根节点
        self.root = BaseCallStackItem()
        # 当前节点
        self.current = self.root
        
    def get_value(self, key: str, default: Any = None) -> Any:
        return self.current.key_values.get(key, default)
    
    def call_next(
        self, 
        key_values: dict[str, Any] = None,
        **kwargs,
    ) -> None:
        # 创建一个调用栈的节点
        item = BaseCallStackItem(prev=self.current, key_values=key_values, **kwargs)
        # 更新当前节点的 next 引用
        self.current.next = item
        # 更新当前节点
        self.current = item
    
    def return_prev(self) -> None:
        # 获取当前节点
        current = self.current
        # 更新当前节点
        self.current = current.prev
        # 解除 prev 的引用
        current.prev.next = None
        # 解除 next 和 prev 的引用
        current.next = None
        current.prev = None
