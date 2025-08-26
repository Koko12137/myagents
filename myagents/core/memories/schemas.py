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
        created_at (int): 创建时间
    """
    id: int = Field(description="ID", default=None)
    
    env_id: str = Field(description="环境ID")
    agent_id: str = Field(description="代理ID")
    task_id: str = Field(description="任务ID")
    memory_id: str = Field(description="记忆ID")
    task_status: str = Field(description="任务状态")
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
    
    def get_embedding(self) -> list[float]:
        return self.embedding
    
    def to_dict(self) -> dict:
        return self.model_dump()


class EpisodeMetadata(BaseModel):
    """情节记忆元数据
    
    属性：
        instruction (str): 需要做什么事
        situation (str): 当前的情况
        action (str): 采取的行动
        result (str): 行动的结果
        reflection (str): 结果的反思
    """
    instruction: str = Field(description="需要做什么事")
    situation: str = Field(description="当前的情况")
    action: str = Field(description="采取的行动")
    result: str = Field(description="行动的结果")
    reflection: str = Field(description="结果的反思")


class EpisodeMemoryItem(BaseVectorMemoryItem):
    """情节记忆数据结构
    
    属性：
        keywords (list[str]): 关键词
        abstract (str): 摘要
        metadata (EpisodeMetadata): 记忆元数据
        is_error (bool): 是否为错误经验
    """
    keywords: list[str] = Field(description="关键词")
    abstract: str = Field(description="摘要")
    metadata: EpisodeMetadata = Field(description="记忆元数据")
    is_error: bool = Field(description="是否为错误经验")
    
    def format(self) -> Union[str, list[dict]]:
        return EPISODE_ITEM_FORMAT.format(
            memory_id=self.memory_id, 
            abstract=self.abstract, 
            keywords=self.keywords, 
            instruction=self.metadata.instruction, 
            situation=self.metadata.situation, 
            action=self.metadata.action, 
            result=self.metadata.result, 
            reflection=self.metadata.reflection, 
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
