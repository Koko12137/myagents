from enum import Enum
from typing import Union

from pydantic import BaseModel, Field

from myagents.prompts.memories.template import EPISODE_ITEM_FORMAT


class MemoryType(Enum):
    """记忆类型枚举"""
    EPISODE = "episode"


class BaseVectorMemoryItem(BaseModel):
    """向量记忆数据结构
    
    属性：
        memory_id (int): 记忆ID
        env_id (str): 环境ID
        agent_id (str): 代理ID
        task_id (str): 任务ID
        metadata (dict): 记忆元数据
        embedding (List[float]): 记忆向量
    """
    id: int = Field(description="ID", default=None)
    
    env_id: str = Field(description="环境ID")
    agent_id: str = Field(description="代理ID")
    task_id: str = Field(description="任务ID")
    memory_id: str = Field(description="记忆ID")
    task_status: str = Field(description="任务状态")
    metadata: dict = Field(description="记忆元数据")
    embedding: list[float] = Field(description="记忆向量")
    created_at: int = Field(description="创建时间")
    
    def get_id(self) -> int:
        return self.id
    
    def get_env_id(self) -> str:
        return self.env_id
    
    def get_agent_id(self) -> str:
        return self.agent_id
    
    def get_task_id(self) -> str:
        return self.task_id
    
    def get_memory_id(self) -> str:
        return self.memory_id
    
    def get_task_status(self) -> str:
        return self.task_status
    
    def get_metadata(self) -> dict:
        return self.metadata
    
    def get_embedding(self) -> list[float]:
        return self.embedding
    
    def to_dict(self) -> dict:
        return self.model_dump()


class EpisodeMemoryItem(BaseVectorMemoryItem):
    """情节记忆数据结构
    
    属性：
        memory_id (int): 记忆ID
        env_id (str): 环境ID
        agent_id (str): 代理ID
        task_id (str): 任务ID
        task_status: str
        metadata (dict): 记忆元数据
        embedding (List[float]): 记忆向量
        abstract (str): 摘要
        keywords (list[str]): 关键词
        is_error (bool): 是否为错误经验
    """
    is_error: bool = Field(description="是否为错误经验")
    
    def format(self) -> Union[str, list[dict]]:
        return EPISODE_ITEM_FORMAT.format(
            memory_id=self.memory_id, 
            abstract=self.metadata.get("abstract"), 
            keywords=self.metadata.get("keywords"), 
            content=self.metadata.get("content"), 
            is_error=self.is_error, 
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
        memory (BaseVectorMemoryItem): 操作的记忆项
    """
    operation: MemoryOperationType = Field(description="操作类型")
    memory: BaseVectorMemoryItem = Field(description="操作的记忆项")
    
    def get_operation(self) -> MemoryOperationType:
        return self.operation
    
    def get_memory(self) -> BaseVectorMemoryItem:
        return self.memory


class MemoryIDMap(BaseModel):
    """记忆ID映射数据结构
    
    属性：
        fake_id (int): 虚拟ID，用于在记忆上下文中使用
        raw_id (int): 原始ID，用于在向量数据库中使用
        memory (BaseVectorMemoryItem): 记忆项
    """
    fake_id: int = Field(description="虚拟ID，用于在记忆上下文中使用")
    raw_id: int = Field(description="原始ID，用于在向量数据库中使用")
    memory: BaseVectorMemoryItem = Field(description="记忆项")
