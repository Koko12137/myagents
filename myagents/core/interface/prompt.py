from abc import abstractmethod
from enum import Enum
from typing import Protocol, runtime_checkable


@runtime_checkable
class Prompt(Protocol):
    """提示词协议，用于存储 Workflow 不同运行阶段的提示词
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """获取提示词的名称"""
    
    @abstractmethod
    def string(self) -> str:
        """获取提示词"""


@runtime_checkable
class PromptGroup(Protocol):
    """提示词组协议，用于管理 Workflow 不同运行阶段的提示词
    """
    
    @abstractmethod
    def get_prompt(self, name: str) -> Prompt:
        """获取提示词
        
        参数:
            name (str):
                提示词的名称
        
        返回:
            Prompt:
                提示词
        
        异常:
            KeyError:
                如果提示词不存在，则抛出 KeyError
        """
    
    @abstractmethod
    def get_sub_group(self, group_name: str) -> 'PromptGroup':
        """获取子提示词组
        
        参数:
            group_name (str):
                子提示词组的名称
        
        返回:
            PromptGroup:
                子提示词组
            
        异常:
            KeyError:
                如果子提示词组不存在，则抛出 KeyError
        """

    @abstractmethod
    def has_prompts(self, names: list[str]) -> bool:
        """检查提示词组是否包含指定的提示词
        
        参数:
            names (list[str]):
                提示词的名称列表
        
        返回:
            bool:
                如果提示词组包含指定的提示词，则返回 True，否则返回 False
        """

    @abstractmethod
    def has_sub_groups(self, group_names: list[str]) -> bool:
        """检查提示词组是否包含指定的子提示词组
        
        参数:
            group_names (list[str]):
                子提示词组的名称列表
        
        返回:
            bool:
                如果提示词组包含指定的子提示词组，则返回 True，否则返回 False
        """
