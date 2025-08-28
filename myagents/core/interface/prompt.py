from abc import abstractmethod
from enum import Enum
from typing import Protocol, runtime_checkable


@runtime_checkable
class Prompt(Protocol):
    """提示词协议，用于存储 Workflow 不同运行阶段的提示词
    """
    
    @abstractmethod
    def get_run_stage(self) -> Enum:
        """获取提示词的运行阶段"""
        pass
    
    @abstractmethod
    def get_prompt(self) -> str:
        """获取提示词"""
        pass


@runtime_checkable
class PromptGroup(Protocol):
    """提示词组协议，用于管理 Workflow 不同运行阶段的提示词
    """
    
    @abstractmethod
    def get_prompt(self, run_stage: Enum) -> Prompt:
        """获取提示词"""
        pass
    
    @abstractmethod
    def get_sub_group(self, group_name: str) -> 'PromptGroup':
        """获取子提示词组"""
        pass
