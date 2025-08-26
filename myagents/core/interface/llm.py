from abc import abstractmethod, ABC
from enum import Enum
from typing import Any, Union

from myagents.core.messages import ToolCallResult, AssistantMessage, UserMessage, SystemMessage


class Queue(ABC):
    """队列的协议
    
    属性:
        queue (Queue):
            队列
    """

    @abstractmethod
    async def put(self, *args, **kwargs) -> None:
        """将项目放入队列
        
        参数:
            *args:
                传递给put方法的额外参数
            **kwargs:
                传递给put方法的额外关键字参数
        """
        pass
    
    @abstractmethod
    async def get(self, *args, **kwargs) -> Any:
        """从队列获取项目
        
        参数:
            *args:
                传递给get方法的额外参数
            **kwargs:
                传递给get方法的额外关键字参数
        """
        pass


class Provider(Enum):
    """语言模型的提供商
    
    - OPENAI (str): OpenAI 提供商
    """
    OPENAI = "openai"
    

class CompletionConfig(ABC):
    """语言模型配置的协议
    
    属性:
        provider (Provider):
            语言模型的提供商
    """

    @abstractmethod
    def to_dict(self, provider: Provider) -> dict:
        """将完成配置转换为字典
        
        参数:
            provider (Provider):
                语言模型的提供商
                
        返回:
            dict:
                作为字典的完成配置
        """
        pass
    
    @abstractmethod
    def update(self, **kwargs) -> None:
        """更新完成配置
        
        参数:
            **kwargs:
                更新完成配置的额外关键字参数
        """
        pass


class LLM(ABC):
    """语言模型的协议
    
    属性:
        provider (Provider):
            语言模型的提供商
        model (str):
            语言模型的模型
        base_url (str):
            语言模型的基础URL
    """
    provider: Provider
    model: str
    base_url: str
    
    @abstractmethod
    async def completion(
        self, 
        messages: list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]], 
        completion_config: CompletionConfig, 
    ) -> AssistantMessage:
        """补全消息
        
        参数:
            messages (list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]):
                要补全的消息
            completion_config (CompletionConfig):
                补全消息配置

        返回:
            AssistantMessage:
                来自语言模型的名为AssistantMessage的完成消息
        """
        pass

    
class EmbeddingLLM(LLM):
    """EmbeddingLLM 是用于嵌入文本的语言模型
    
    属性:
        provider (Provider):
            语言模型的提供商
        model (str):
            语言模型的模型
        base_url (str):
            语言模型的基础URL
    """
    
    @abstractmethod
    async def embed(self, text: str, **kwargs) -> list[float]:
        """嵌入文本并返回嵌入向量
        
        参数:
            text (str):
                要嵌入的文本
                
        返回:
            list[float]:
                文本的嵌入向量
        """
        pass
