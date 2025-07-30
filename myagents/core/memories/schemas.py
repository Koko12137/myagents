from enum import Enum

from pydantic import BaseModel


class MemoryType(Enum):
    """记忆类型枚举"""
    VECTOR = "vector"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class BaseVectorMemory(BaseModel):
    """向量记忆数据结构
    
    属性：
        env_id (str): 环境ID
        agent_id (str): 代理ID
        task_id (str): 任务ID
        memory_id (str): 记忆ID
        content (str): 记忆内容
        embedding (List[float]): 记忆向量
    """
    env_id: str
    agent_id: str
    task_id: str
    memory_id: str
    content: str
    embedding: list[float]
