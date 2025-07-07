from abc import abstractmethod
from enum import Enum
from typing import Union

from myagents.core.interface import Stateful
from myagents.core.messages import AssistantMessage, UserMessage, SystemMessage, ToolCallResult, MessageRole


class StateMixin(Stateful):
    """StateMixin is a mixin class for stateful entity management.
    
    Attributes:
        status (Enum):
            The status of the stateful entity.
        history (dict[str, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]]):
            The history of the stateful entity.
    """
    status: Enum
    history: dict[str, list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]]
    
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
        if isinstance(message, SystemMessage) and len(self.history[self.status]) > 0:
            # Convert the SystemMessage to a UserMessage
            message = UserMessage(content=message.content)
            return self.update(message)
        
        elif isinstance(message, UserMessage) and len(self.history[self.status]) > 0:
            last_message = self.history[self.status][-1]
            if last_message.role == MessageRole.USER:
                last_message.content = f"{last_message.content}\n{message.content}"
                last_message.stop_reason = message.stop_reason
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
        for attr in self.status.__dict__:
            if isinstance(attr, Enum):
                self.history[attr] = []

    def get_status(self) -> Enum:
        """Get the current status of the stateful entity. 
        
        Returns:
            Enum:
                The current status of the stateful entity.
        """
        return self.status
