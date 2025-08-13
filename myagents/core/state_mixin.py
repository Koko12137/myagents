from enum import Enum
from uuid import uuid4
from typing import Union

from myagents.core.interface import Stateful
from myagents.core.messages import AssistantMessage, UserMessage, SystemMessage, ToolCallResult, MessageRole


class StateMixin(Stateful):
    """StateMixin is a mixin class for stateful entity management.
    
    Attributes:
        uid (str):
            The unique identifier of the stateful entity.
        status_class (Enum):
            The status class of the stateful entity.
        status (Enum):
            The status of the stateful entity.
        history (dict[str, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]]):
            The history of the stateful entity.
    """
    uid: str
    status_class: Enum
    status: Enum
    history: dict[str, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]]
    
    def __init__(self, status_class: Enum, *args, **kwargs) -> None:
        """Initialize the StateMixin.
        
        Args:
            status_class (Enum):
                The status class of the stateful entity.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        super().__init__(*args, **kwargs)
        # Initialize the uid
        self.uid = uuid4().hex
        # Initialize the status class
        self.status_class = status_class
        # Initialize the history
        self.history = {}
        # Reset the history
        self.reset()
    
    def update(
        self, 
        message: Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult], 
    ) -> None:
        """Update the task status. If the last message is the same role as the current message, the content will be 
        concatenated. Otherwise, the message will be appended directly. 
        
        Args:
            message (Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]):
                The message to be updated.
        
        Returns:
            None
        """
        if isinstance(message, SystemMessage):
            if len(self.history[self.status]) > 0:
                last_message = self.history[self.status][-1]
                if last_message.role == MessageRole.SYSTEM:
                    # 如果前一个也是SystemMessage，则追加内容
                    last_message.content = f"{last_message.content}\n=====[Message Block]=====\n{message.content}"
                else:
                    # 如果前一个不是SystemMessage，则转换为UserMessage
                    message = UserMessage(content=message.content)
                    return self.update(message)
            else:
                self.history[self.status].append(message)
        
        elif isinstance(message, UserMessage):
            if len(self.history[self.status]) > 0:
                # 获取最后一个消息
                last_message = self.history[self.status][-1]
                if last_message.role == MessageRole.USER:
                    # 如果前一个也是UserMessage，则追加内容
                    last_message.content = f"{last_message.content}\n=====[Message Block]=====\n{message.content}"
                else:
                    # 如果前一个不是UserMessage，则直接添加
                    self.history[self.status].append(message)
            else:
                self.history[self.status].append(message)
        
        elif isinstance(message, AssistantMessage):
            if len(self.history[self.status]) == 0:
                raise ValueError("The history is empty, but the message is an assistant message.")
            elif self.history[self.status][-1].role == MessageRole.ASSISTANT:
                raise ValueError("The last message is an assistant message, but another new assistant message is coming.")
            else:
                self.history[self.status].append(message)
        
        elif isinstance(message, ToolCallResult):
            if len(self.history[self.status]) == 0:
                raise ValueError("The history is empty, but the message is a tool call result.")
            elif self.history[self.status][-1].role == MessageRole.USER:
                raise ValueError("The last message is a user message, but a tool call result is coming.")
            else:
                self.history[self.status].append(message)
            
        else:
            raise ValueError(f"The message is not a valid message: {type(message)}")
            
    def get_history(self) -> list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]:
        """Get the history of the stateful entity according to the current status.
                
        Returns:
            list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]:
                The history of the stateful entity according to the current status.
        """
        # Return the history of the status
        return self.history[self.status]
    
    def reset(self) -> None:
        """Reset the history for all the statuses.
        """
        # Traverse all the attributes of the status class
        for attr in self.status_class.__dict__:
            if isinstance(self.status_class.__dict__[attr], Enum):
                self.history[self.status_class.__dict__[attr]] = []

    def get_status(self) -> Enum:
        """Get the current status of the stateful entity. 
        
        Returns:
            Enum:
                The current status of the stateful entity.
        """
        return self.status
