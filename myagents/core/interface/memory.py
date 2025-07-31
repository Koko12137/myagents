from abc import abstractmethod
from typing import runtime_checkable, Protocol


@runtime_checkable
class VectorMemory(Protocol):
    """向量记忆管理的协议。用于存储代理在任何有状态实体上工作的向量记忆
    """
    
    @abstractmethod
    async def add(self, memories: list, **kwargs) -> None:
        """将文本添加到记忆中
        
        参数:
            memories (list):
                要添加到记忆中的记忆列表
        """
        pass
    
    @abstractmethod
    async def search(
        self, 
        query_embedding: list[float], 
        top_k: int, 
        score_threshold: float, 
        condition: str, 
        **kwargs,
    ) -> list[tuple[dict, float]]:
        """从记忆中搜索相关文本
        
        参数:
            query_embedding (list[float]): 
                搜索相关文本的查询向量
            top_k (int): 
                返回的记忆数量
            score_threshold (float): 
                记忆分数的阈值
            condition (str):
                搜索条件
        返回:
            list[tuple[dict, float]]:
                搜索到的记忆列表，每个记忆包含记忆ID、记忆内容和记忆分数
        """
        pass
    
    @abstractmethod
    async def update(
        self, 
        memories: list, 
        **kwargs,
    ) -> None:
        """更新记忆中的文本
        
        参数:
            memories (list): 
                要更新的记忆列表
        """
        pass
    
    @abstractmethod
    async def delete(
        self, 
        memory_ids: list[int], 
        **kwargs,
    ) -> None:
        """从记忆中删除文本
        
        参数:
            memory_ids (list[int]): 
                要删除的记忆ID列表
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
