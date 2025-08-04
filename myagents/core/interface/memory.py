from abc import abstractmethod
from enum import Enum
from typing import runtime_checkable, Protocol, Union


@runtime_checkable
class VectorMemoryItem(Protocol):
    """向量记忆项的协议。向量数据库里存储的最小单元
    """
    
    @abstractmethod
    def get_memory_id(self) -> int:
        """获取向量记忆项的ID。向量记忆项的ID是向量数据库中存储的唯一标识符。
        
        返回:
            int: 向量记忆项的ID
        """
        pass
    
    @abstractmethod
    def get_agent_id(self) -> int:
        """获取向量记忆项的代理ID。向量记忆项的代理ID是向量数据库中存储的代理ID。
        
        返回:
            int: 向量记忆项的代理ID
        """
        pass
    
    @abstractmethod
    def get_task_id(self) -> int:
        """获取向量记忆项的任务ID。向量记忆项的任务ID是向量数据库中存储的任务ID。
        
        返回:
            int: 向量记忆项的任务ID
        """
        pass
    
    @abstractmethod
    def get_task_status(self) -> str:
        """获取向量记忆项的任务状态。向量记忆项的任务状态是向量数据库中存储的任务状态。
        
        返回:
            str: 向量记忆项的任务状态
        """
        pass
    
    @abstractmethod
    def get_content(self) -> str:
        """获取向量记忆项的内容。向量记忆项的内容是向量数据库中存储的内容。
        
        返回:
            str: 向量记忆项的内容
        """
        pass
    
    @abstractmethod
    def get_embedding(self) -> list[float]:
        """获取向量记忆项的嵌入向量。
        
        返回:
            list[float]: 向量记忆项的嵌入向量
        """
        pass
    
    @abstractmethod
    def format(self, **kwargs) -> Union[str, list[dict]]:
        """将向量记忆项格式化为字符串。
        
        参数:
            **kwargs:
                格式化参数
        
        返回:
            Union[str, list[dict]]: 格式化后的向量记忆项
        """
        pass
    
    @abstractmethod
    def to_dict(self) -> dict:
        """将向量记忆项转换为字典。
        
        返回:
            dict: 向量记忆项的字典
        """
        pass
    
    
@runtime_checkable
class MemoryOperation(Protocol):
    """记忆操作的协议。用于存储代理在任何有状态实体上工作的记忆操作
    """
    
    @abstractmethod
    def get_operation(self) -> Enum:
        """获取记忆操作的类型。记忆操作的类型是向量数据库中存储的类型。
        
        返回:
            str: 记忆操作的类型
        """
        pass
    
    @abstractmethod
    def get_memory(self) -> VectorMemoryItem:
        """获取记忆操作的记忆。记忆操作的记忆是向量数据库中存储的记忆。
        
        返回:
            VectorMemoryItem: 记忆操作的记忆
        """
        pass


@runtime_checkable
class VectorMemoryCollection(Protocol):
    """向量记忆管理的协议。用于存储代理在任何有状态实体上工作的向量记忆
    """
    
    @abstractmethod
    def get_dimension(self) -> int:
        """获取向量记忆项的维度。向量记忆项的维度是向量数据库中存储的维度。
        
        返回:
            int: 向量记忆项的维度
        """
        pass
    
    @abstractmethod
    def get_index_type(self) -> str:
        """获取向量记忆项的索引类型。向量记忆项的索引类型是向量数据库中存储的索引类型。
        
        返回:
            str: 向量记忆项的索引类型
        """
        pass
    
    @abstractmethod
    def get_metric_type(self) -> str:
        """获取向量记忆项的距离度量类型。向量记忆项的距离度量类型是向量数据库中存储的距离度量类型。
        
        返回:
            str: 向量记忆项的距离度量类型
        """
        pass
    
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
class TableMemoryDB(Protocol):
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
