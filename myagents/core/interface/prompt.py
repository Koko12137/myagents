from abc import abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class Prompt(Protocol):
    """提示词协议，需要包括：
    1. 任务要求
    2. 格式要求
    3. Few-Shots
    4. 记忆
    5. 工具
    6. 其他要求
    """
    
    @abstractmethod
    def get_task_requirements(self) -> str:
        """获取任务要求"""
        pass
    
    @abstractmethod
    def set_task_requirements(self, task_requirements: str) -> None:
        """设置任务要求"""
        pass
    
    @abstractmethod
    def get_format_requirements(self) -> str:
        """获取格式要求"""
        pass
    
    @abstractmethod
    def set_format_requirements(self, format_requirements: str) -> None:
        """设置格式要求"""
        pass
    
    @abstractmethod
    def get_few_shots(self) -> list[str]:
        """获取 Few-Shots"""
        pass
    
    @abstractmethod
    def set_few_shots(self, few_shots: list[str]) -> None:
        """设置 Few-Shots"""
        pass
    
    @abstractmethod
    def get_memory(self) -> list[str]:
        """获取记忆"""
        pass
    
    @abstractmethod
    def set_memory(self, memory: list[str]) -> None:
        """设置记忆"""
        pass
    
    @abstractmethod
    def get_tools(self) -> list[str]:
        """获取工具"""
        pass
    
    @abstractmethod
    def set_tools(self, tools: list[str]) -> None:
        """设置工具"""
        pass
    
    @abstractmethod
    def get_other_requirements(self) -> str:
        """获取其他要求"""
        pass
    
    @abstractmethod
    def set_other_requirements(self, other_requirements: str) -> None:
        """设置其他要求"""
        pass
    
    @abstractmethod
    def get_prompt(self) -> str:
        """获取提示词"""
        pass
