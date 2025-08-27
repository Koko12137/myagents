from typing import Any, Optional
import time
import uuid

from loguru import logger
from pymilvus import (
    AsyncMilvusClient, 
    CollectionSchema, 
    FieldSchema, 
    DataType,
    MilvusException,
)

from myagents.schemas.dbs.milvus import (
    EpisodeMemoryItem, 
    EpisodeMetadata, 
    MemoryType,
)


class MilvusEpisodeMemoryCollection:
    """基于Milvus的情节记忆管理类
    
    属性:
        client (AsyncMilvusClient):
            Milvus客户端
        collection_name (str):
            集合名称
        dimension (int, defaults to 1024):
            向量维度
        index_type (str, defaults to "IVF_FLAT"):
            索引类型
        metric_type (str, defaults to "COSINE"):
            距离度量类型
        valid_memory_types (list[MemoryType]):
            合法的记忆类型，包括:
                MemoryType.EPISODE: 情节(事件、经历、经验)记忆
    """
    client: AsyncMilvusClient
    collection_name: str
    dimension: int
    index_type: str
    metric_type: str
    valid_memory_types: list[MemoryType] = [
        MemoryType.EPISODE, 
    ]
    
    def __init__(
        self, 
        client: AsyncMilvusClient, 
        collection_name: str, 
        dimension: int = 1024, 
        index_type: str = "IVF_FLAT", 
        metric_type: str = "COSINE", 
        **kwargs, 
    ):
        """
        初始化向量记忆管理器
        
        Args:
            client: Milvus客户端
            collection_name: 集合名称
            dimension: 向量维度
            index_type: 索引类型
            metric_type: 距离度量类型
            **kwargs: 其他参数
        """
        self.client = client
        self.collection_name = collection_name
        self.dimension = dimension
        self.index_type = index_type
        self.metric_type = metric_type
        
    def get_dimension(self) -> int:
        """获取向量记忆项的维度。向量记忆项的维度是向量数据库中存储的维度。
        
        返回:
            int: 向量记忆项的维度
        """
        return self.dimension
    
    def get_index_type(self) -> str:
        """获取向量记忆项的索引类型。向量记忆项的索引类型是向量数据库中存储的索引类型。
        
        返回:
            str: 向量记忆项的索引类型
        """
        return self.index_type
    
    def get_metric_type(self) -> str:
        """获取向量记忆项的距离度量类型。向量记忆项的距离度量类型是向量数据库中存储的距离度量类型。
        
        返回:
            str: 向量记忆项的距离度量类型
        """
        return self.metric_type
    
    async def add(self, memories: list[EpisodeMemoryItem]) -> None:
        """插入向量记忆"""
        # 准备数据
        data = [memory.model_dump() for memory in memories]
        # Drop the memory_id field
        for memory in data:
            memory.pop("id", None)
            # 检查是否存在 memory_id，不存在则添加
            if "memory_id" not in memory:
                memory["memory_id"] = uuid.uuid4().hex
            # 检查是否存在 created_at，不存在则添加
            if "created_at" not in memory:
                memory["created_at"] = int(time.time())
        
        # 插入数据
        await self.client.insert(self.collection_name, data)
        
        logger.info(f"成功插入向量记忆: {len(memories)}")
    
    async def search(
        self, 
        env_id: str,
        agent_id: str,
        task_id: str,
        task_status: str,
        query_embedding: list[float], 
        top_k: int = 10,
        score_threshold: float = 0.5, 
        condition: str = None, 
    ) -> list[tuple[dict, float]]:
        """搜索相似向量记忆"""
        try:
            # 查询条件
            expr = f'env_id == "{env_id}" AND agent_id == "{agent_id}" AND task_id == "{task_id}" AND task_status == "{task_status}"'
            if condition is not None:
                expr += f' AND {condition}'
            
            # 执行搜索
            results = await self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                anns_field="embedding",
                limit=top_k,
                output_fields=[
                    "id", "env_id", "agent_id", "task_id", "memory_id", "task_status", 
                    "created_at", "embedding", "keywords", "abstract", "is_error", "metadata",
                ],
                filter=expr,
            )
            
            # 解析结果
            memories: list[tuple[dict, float]] = []
            for hits in results:
                for hit in hits:
                    if hit.score >= score_threshold:
                        # 合法性检查
                        memory = EpisodeMemoryItem(
                            id=hit.get("id"),
                            env_id=hit.get("env_id"),
                            agent_id=hit.get("agent_id"),
                            task_id=hit.get("task_id"),
                            memory_id=hit.get("memory_id"),
                            task_status=hit.get("task_status"),
                            embedding=hit.get("embedding"),
                            created_at=hit.get("created_at"),
                            keywords=hit.get("keywords"),
                            abstract=hit.get("abstract"),
                            is_error=hit.get("is_error"),
                            metadata=EpisodeMetadata(**hit.get("metadata")),
                        )
                        
                        # 兜底检查
                        if (memory.env_id != env_id or 
                            memory.agent_id != agent_id or 
                            memory.task_status != task_status or 
                            memory.task_id != task_id
                        ):
                            continue
                        
                        # 添加到记忆列表
                        memories.append((memory.model_dump(), hit.score))
            
            return memories
            
        except Exception as e:
            logger.error(f"搜索向量记忆失败: {e}")
            return []
    
    async def get_collection_stats(self) -> dict[str, Any]:
        """获取集合统计信息"""
        try:
            return await self.client.describe_collection(self.collection_name)
            
        except Exception as e:
            logger.error(f"获取集合统计信息失败: {e}")
            return {}


class MilvusManager:
    """Milvus连接和集合管理类 - 支持标准Milvus和Milvus Lite"""
    
    def __init__(
        self,
        url: str = None, 
        host: str = None,
        port: str = None,
        user: str = "",
        password: str = "",
        db_name: str = "default",
    ):
        """
        初始化Milvus管理器
        
        Args:
            host: Milvus服务器地址
            port: Milvus服务器端口
            user: 用户名
            password: 密码
            db_name: 数据库名称
            use_lite: 是否使用Milvus Lite本地数据库
            lite_data_path: Milvus Lite数据存储路径
        """
        if url is None:
            if host is None and port is None:
                # 使用本地 Milvus Lite
                url = "./milvus_demo.db"
            else:
                if host is None:
                    host = "localhost"
                if port is None:
                    port = "19530"
                url = f"http://{host}:{port}"
        else:
            if host is not None or port is not None:
                raise ValueError("url和host、port不能同时提供")
        
        self.url = url
        self.host = host
        self.port = port
        self.db_name = db_name 
        self.loaded_collections: dict[str, MilvusEpisodeMemoryCollection] = {}
        
        # 连接数据库
        self.client = AsyncMilvusClient(
            uri=self.url, 
            user=user, 
            password=password, 
            db_name=self.db_name, 
        )
    
    async def create_episode_memory(
        self,
        collection_name: str,
        dimension: int = 1536,
        index_type: str = "IVF_FLAT",
        metric_type: str = "COSINE", 
        **kwargs,
    ) -> Optional[MilvusEpisodeMemoryCollection]:
        """创建向量记忆管理器"""
        try:
            # 定义字段模式
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="env_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="agent_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="task_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="memory_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="task_status", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="created_at", dtype=DataType.INT64),
                FieldSchema(name="keywords", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="is_error", dtype=DataType.BOOL),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]
            # 创建集合模式
            schema = CollectionSchema(fields=fields, description=f"Vector memory collection: {collection_name}")
            
            # 创建索引参数
            index_params = self.client.prepare_index_params(
                field_name="embedding", 
            )
            index_params.add_index(
                field_name="embedding", 
                index_type=index_type, 
                metric_type=metric_type, 
            )
            
            # 创建集合
            await self.client.create_collection(
                collection_name=collection_name, 
                dimension=dimension, 
                schema=schema, 
                index_params=index_params, 
            )
            
            # 创建新的向量记忆管理器
            memory = MilvusEpisodeMemoryCollection(
                collection_name=collection_name, 
                client=self.client, 
                dimension=dimension, 
                index_type=index_type, 
                metric_type=metric_type, 
                **kwargs,
            )
            
            # 添加到已加载的集合中
            self.loaded_collections[collection_name] = memory
            logger.info(f"成功创建向量记忆管理器: {collection_name}")
            return memory
                
        except MilvusException as e:
            logger.error(f"创建向量记忆管理器失败: {e}")
            raise e
    
    async def drop_vector_memory(self, collection_name: str) -> bool:
        """删除向量记忆管理器"""
        try:
            # 卸载集合
            await self.client.release_collection(collection_name)
            # 删除集合
            await self.client.drop_collection(collection_name)
            del self.loaded_collections[collection_name]
            return True
                
        except Exception as e:
            logger.error(f"删除向量记忆管理器失败: {e}")
            return False

    async def get_collection(
        self, 
        collection_name: str, 
        dimension: int = 1536,
        index_type: str = "IVF_FLAT",
        metric_type: str = "COSINE", 
        **kwargs,
    ) -> MilvusEpisodeMemoryCollection:
        """获取向量记忆管理器
        
        参数:
            collection_name: 集合名称
            **kwargs: 其他参数
            
        返回:
            MilvusMemoryCollection: 向量记忆管理器
        """
        try:
            if collection_name not in self.loaded_collections:
                # 加载集合
                await self.client.load_collection(collection_name)
                # 创建新的向量记忆管理器
                memory = MilvusEpisodeMemoryCollection(
                    collection_name=collection_name, 
                    client=self.client, 
                    dimension=dimension,
                    index_type=index_type,
                    metric_type=metric_type,
                    **kwargs,
                )
                # 添加到已加载的集合中
                self.loaded_collections[collection_name] = memory
                return memory
            else:
                # 返回已加载的集合
                return self.loaded_collections[collection_name]
        except MilvusException as e:
            logger.error(f"获取向量记忆管理器失败: {e}，尝试创建新的向量记忆管理器")
            return await self.create_episode_memory(
                collection_name=collection_name, 
                dimension=dimension, 
                index_type=index_type, 
                metric_type=metric_type, 
                **kwargs,
            )
