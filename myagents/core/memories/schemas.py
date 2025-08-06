from enum import Enum
from typing import Union

from pydantic import BaseModel, Field

from myagents.prompts.workflows.memory import SEMANTIC_ITEM_FORMAT, EPISODE_ITEM_FORMAT


class MemoryType(Enum):
    """记忆类型枚举"""
    SEMANTIC = "semantic_memory"
    EPISODE = "episode_memory"


class BaseVectorMemoryItem(BaseModel):
    """向量记忆数据结构
    
    属性：
        memory_type (str): 记忆类型
        memory_id (int): 记忆ID
        env_id (int): 环境ID
        agent_id (int): 代理ID
        task_id (int): 任务ID
        content (str): 记忆内容
        embedding (List[float]): 记忆向量
    """
    memory_type: str = Field(description="记忆类型")
    memory_id: int = Field(description="记忆ID")
    env_id: int = Field(description="环境ID")
    agent_id: int = Field(description="代理ID")
    task_id: int = Field(description="任务ID")
    task_status: str = Field(description="任务状态")
    content: str = Field(description="记忆内容")
    embedding: list[float] = Field(description="记忆向量")
    
    def get_memory_id(self) -> int:
        return self.memory_id
    
    def get_memory_type(self) -> MemoryType:
        return MemoryType(self.memory_type)
    
    def get_env_id(self) -> int:
        return self.env_id
    
    def get_agent_id(self) -> int:
        return self.agent_id
    
    def get_task_id(self) -> int:
        return self.task_id
    
    def get_task_status(self) -> str:
        return self.task_status
    
    def get_content(self) -> str:
        return self.content
    
    def get_embedding(self) -> list[float]:
        return self.embedding
    
    def to_dict(self) -> dict:
        return {
            "memory_type": self.memory_type.value,
            "memory_id": self.memory_id,
            "env_id": self.env_id,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "task_status": self.task_status,
            "content": self.content,
            "embedding": self.embedding,
        }


class SemanticMemoryItem(BaseVectorMemoryItem):
    """语义记忆数据结构
    
    属性：
        memory_type (str): 记忆类型
        memory_id (int): 记忆ID
        env_id (int): 环境ID
        agent_id (int): 代理ID
        task_id (int): 任务ID
        task_status: str
        content (str): 记忆内容
        embedding (List[float]): 记忆向量
        truth_value (bool): 记忆真假
    """
    truth_value: bool = Field(description="记忆真假")
        
    def get_truth_value(self) -> bool:
        return self.truth_value
    
    def format(self) -> Union[str, list[dict]]:
        return SEMANTIC_ITEM_FORMAT.format(
            memory_id=self.memory_id, 
            content=self.content, 
            truth_value=self.truth_value
        )


class EpisodeMemoryItem(BaseVectorMemoryItem):
    """情节记忆数据结构
    
    属性：
        memory_type (str): 记忆类型
        memory_id (int): 记忆ID
        env_id (int): 环境ID
        agent_id (int): 代理ID
        task_id (int): 任务ID
        task_status: str
        content (str): 记忆内容
        embedding (List[float]): 记忆向量
        positive_impact (bool): 记忆积极与否
    """
    positive_impact: bool = Field(description="记忆积极与否")
        
    def get_positive_impact(self) -> bool:
        return self.positive_impact
    
    def format(self) -> Union[str, list[dict]]:
        return EPISODE_ITEM_FORMAT.format(
            memory_id=self.memory_id, 
            content=self.content, 
            positive_impact=self.positive_impact
        )
    
class MemoryOperationType(Enum):
    """记忆操作类型枚举
    
    属性：
        ADD: 添加记忆
        UPDATE: 更新记忆
        DELETE: 删除记忆
    """
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"

class BaseMemoryOperation(BaseModel):
    """记忆操作数据结构
    
    属性：
        operation (MemoryOperationType): 操作类型
        memory (BaseVectorMemory): 操作的记忆
    """
    operation: MemoryOperationType = Field(description="操作类型")
    memory: Union[SemanticMemoryItem, EpisodeMemoryItem] = Field(description="操作的记忆")
    
    def get_operation(self) -> MemoryOperationType:
        return self.operation
    
    def get_memory(self) -> Union[SemanticMemoryItem, EpisodeMemoryItem]:
        return self.memory
