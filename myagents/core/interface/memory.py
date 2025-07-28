from abc import abstractmethod
from typing import runtime_checkable, Protocol

from myagents.core.interface.llm import EmbeddingLLM


@runtime_checkable
class VectorMemory(Protocol):
    """向量记忆管理的协议。用于存储代理在任何有状态实体上工作的向量记忆
    
    参数:
        embedding_llm (EmbeddingLLM):
            用于嵌入文本的嵌入语言模型
    """
    embedding_llm: EmbeddingLLM
    
    @abstractmethod
    async def add(
        self, 
        text: str, 
        **kwargs,
    ) -> None:
        """将文本添加到记忆中
        
        参数:
            text (str):
                要添加到记忆中的文本
        """
        pass
    
    @abstractmethod
    async def search(
        self, 
        query: str, 
        **kwargs,
    ) -> list[str]:
        """从记忆中搜索相关文本
        
        参数:
            query (str): 
                搜索相关文本的查询
        """
        pass
    
    @abstractmethod
    async def update(
        self, 
        memory_id: str, 
        text: str, 
        **kwargs,
    ) -> None:
        """更新记忆中的文本
        
        参数:
            memory_id (str): 
                要更新的记忆ID
            text (str): 
                要在记忆中更新的文本
        """
        pass
    
    @abstractmethod
    async def delete(
        self, 
        memory_id: str, 
        **kwargs,
    ) -> None:
        """从记忆中删除文本
        
        参数:
            memory_id (str): 
                要删除的记忆ID
        """
        pass


@runtime_checkable
class TableMemory(Protocol):
    """表记忆管理的协议。用于存储代理在任何有状态实体上工作的表记忆
    """
    
    @abstractmethod
    async def add(
        self, 
        data: dict, 
        **kwargs,
    ) -> None:
        """将数据添加到记忆中
        
        参数:
            data (dict):
                要添加到记忆中的数据
        """
        pass

    @abstractmethod
    async def search(
        self, 
        memory_id: str, 
        **kwargs,
    ) -> list[dict]:
        """从记忆中搜索相关数据
        
        参数:
            memory_id (str):
                要搜索的记忆ID
        """
        pass
    
    @abstractmethod
    async def update(
        self, 
        memory_id: str, 
        data: dict, 
        **kwargs,
    ) -> None:
        """更新记忆中的数据
        
        参数:
            memory_id (str): 
                要更新的记忆ID
            data (dict):
                要在记忆中更新的数据
        """
        pass
    
    @abstractmethod
    async def delete(
        self, 
        memory_id: str, 
        **kwargs,
    ) -> None:
        """从记忆中删除数据
        
        参数:
            memory_id (str): 
                要删除的记忆ID
        """
        pass
