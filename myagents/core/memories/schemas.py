from enum import Enum
from uuid import uuid4
from typing import Union

from pydantic import BaseModel, Field, field_validator


class MemoryType(Enum):
    """记忆类型枚举"""
    SEMANTIC = "semantic_memory"
    EPISODE = "episode_memory"
    PROCEDURAL = "procedural_memory"


class BaseVectorMemory(BaseModel):
    """向量记忆数据结构
    
    属性：
        memory_type (MemoryType): 记忆类型
        memory_id (int): 记忆ID, 默认使用 uuid4 生成
        env_id (int): 环境ID
        agent_id (int): 代理ID
        task_id (int): 任务ID
        content (str): 记忆内容
        embedding (List[float]): 记忆向量
    """
    memory_type: MemoryType = Field(description="记忆类型")
    memory_id: int = Field(description="记忆ID", default_factory=lambda: uuid4().int)
    env_id: int = Field(description="环境ID")
    agent_id: int = Field(description="代理ID")
    task_id: int = Field(description="任务ID")
    content: str = Field(description="记忆内容")
    embedding: list[float] = Field(description="记忆向量")
    
    
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

class MemoryOperation(BaseModel):
    """记忆操作数据结构
    
    属性：
        operation (MemoryOperationType): 操作类型
        before (BaseVectorMemory): 操作前的记忆, 当 operation 为 DELETE 时必须提供
        after (BaseVectorMemory): 操作后的记忆, 当 operation 为 ADD 时必须提供
    """
    operation: MemoryOperationType = Field(description="操作类型")
    before: BaseVectorMemory = Field(description="操作前的记忆", default=None)
    after: BaseVectorMemory = Field(description="操作后的记忆", default=None)
    
    # 验证是否提供了必要的记忆
    @field_validator("before", "after", pre=True, always=True)
    def validate_memory(cls, v: dict[str, Union[MemoryOperationType, BaseVectorMemory]]) -> BaseVectorMemory:
        if v["operation"] == MemoryOperationType.ADD:
            assert v["after"] is not None, "添加记忆时必须提供操作后的记忆"
        elif v["operation"] == MemoryOperationType.UPDATE:
            assert v["before"] is not None and v["after"] is not None, "更新记忆时必须提供操作前/后的记忆"
        elif v["operation"] == MemoryOperationType.DELETE:
            assert v["before"] is not None, "删除记忆时必须提供操作前的记忆"
        else:
            raise ValueError(f"不支持的操作类型: {v['operation']}")
        return v
    
    
class SemanticMemory(BaseVectorMemory):
    """语义记忆数据结构
    
    属性：
        memory_type (MemoryType): 记忆类型
        memory_id (int): 记忆ID
        env_id (int): 环境ID
        agent_id (int): 代理ID
        task_id (int): 任务ID
        content (str): 记忆内容
        embedding (List[float]): 记忆向量
        truth_value (bool): 记忆真假
    """
    truth_value: bool = Field(description="记忆真假")


class EpisodeMemory(BaseVectorMemory):
    """情节记忆数据结构
    
    属性：
        memory_type (MemoryType): 记忆类型
        memory_id (int): 记忆ID
        env_id (int): 环境ID
        agent_id (int): 代理ID
        task_id (int): 任务ID
        content (str): 记忆内容
        embedding (List[float]): 记忆向量
        positive_impact (bool): 记忆积极与否
    """
    positive_impact: bool = Field(description="记忆积极与否", default=None)


class ProceduralMemory(BaseVectorMemory):
    """程序性记忆数据结构
    
    属性：
        memory_type (MemoryType): 记忆类型
        memory_id (int): 记忆ID
        env_id (int): 环境ID
        agent_id (int): 代理ID
        task_id (int): 任务ID
        content (str): 记忆内容
        embedding (List[float]): 记忆向量
        what (str): 做什么
        how (str): 怎么做
        why (str): 为什么做
        whynot (str): 为什么不做
    """
    what: str = Field(description="做什么", default="")
    how: str = Field(description="怎么做", default="")
    why: str = Field(description="为什么做", default="")
    whynot: str = Field(description="为什么不做", default="")
    