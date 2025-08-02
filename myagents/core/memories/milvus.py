from typing import Any, Optional

from loguru import logger
from pymilvus import (
    AsyncMilvusClient, 
    CollectionSchema, 
    FieldSchema, 
    DataType,
)

from myagents.core.memories.schemas import BaseVectorMemoryItem, BaseMemoryOperation, MemoryOperationType, MemoryType


class MilvusMemoryCollection:
    """基于Milvus的向量记忆管理类
    
    属性:
        client (AsyncMilvusClient):
            Milvus客户端
        collection_name (str):
            集合名称
        valid_memory_types (list[MemoryType]):
            合法的记忆类型，包括:
                MemoryType.SEMANTIC: 语义(事实、知识、信息、数据)记忆
                MemoryType.EPISODE: 情节(事件、经历、经验)记忆
                MemoryType.PROCEDURAL: 程序性(命令、需要做的、需要思考的)记忆
    """
    client: AsyncMilvusClient
    collection_name: str
    valid_memory_types: list[MemoryType] = [
        MemoryType.SEMANTIC, 
        MemoryType.EPISODE, 
        MemoryType.PROCEDURAL, 
    ]
    
    def __init__(
        self, 
        client: AsyncMilvusClient, 
        collection_name: str, 
        **kwargs, 
    ):
        """
        初始化向量记忆管理器
        
        Args:
            client: Milvus客户端
            collection_name: 集合名称
            **kwargs: 其他参数
        """
        self.client = client
        self.collection_name = collection_name
        
    async def operate(self, operation: BaseMemoryOperation) -> bool:
        """执行记忆操作"""
        if operation.operation == MemoryOperationType.ADD:
            return await self.add(operation.after)
        elif operation.operation == MemoryOperationType.UPDATE:
            return await self.update(operation.before, operation.after)
        elif operation.operation == MemoryOperationType.DELETE:
            return await self.delete(operation.before)
        else:
            raise ValueError(f"不支持的操作类型: {operation.operation}")
    
    async def add(self, memories: list[BaseVectorMemoryItem]) -> bool:
        """插入向量记忆"""
        try:
            # 准备数据
            data = [memory.model_dump() for memory in memories]
            
            # 插入数据
            await self.client.insert(self.collection_name, data)
            
            logger.info(f"成功插入向量记忆: {len(memories)}")
            return True
            
        except Exception as e:
            logger.error(f"插入向量记忆失败: {e}")
            return False
    
    async def search(
        self, 
        env_id: str,
        agent_id: str,
        task_id: str,
        query_embedding: list[float], 
        top_k: int = 10,
        score_threshold: float = 0.5, 
        condition: str = None, 
    ) -> list[tuple[dict, float]]:
        """搜索相似向量记忆"""
        try:
            # 查询条件
            expr = f"env_id == {env_id} and agent_id == {agent_id} and task_id == {task_id}"
            if condition is not None:
                expr += f" and {condition}"
            
            # 执行搜索
            results = await self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                anns_field="embedding",
                limit=top_k,
                output_fields=[
                    "env_id", "agent_id", "task_id", "memory_id", "content", "embedding", 
                ],
                filter=expr,
                search_params={"nprobe": 10}, 
            )
            
            # 解析结果
            memories = []
            for hits in results:
                for hit in hits:
                    if hit.score >= score_threshold:
                        # 合法性检查
                        memory = BaseVectorMemoryItem(
                            memory_id=hit.get("memory_id"),
                            env_id=hit.get("env_id"),
                            agent_id=hit.get("agent_id"),
                            task_id=hit.get("task_id"),
                            content=hit.get("content"),
                            embedding=hit.get("embedding"),
                        )
                        memories.append((memory.model_dump(), hit.score))
            
            return memories
            
        except Exception as e:
            logger.error(f"搜索向量记忆失败: {e}")
            return []
    
    async def update(self, memories: list[BaseVectorMemoryItem]) -> bool:
        """更新向量记忆"""
        try:
            # 删除旧记录
            memory_ids = [memory.memory_id for memory in memories]
            await self.client.delete(self.collection_name, f'memory_id in {memory_ids}')
            # 插入新记录
            await self.add(memories)
            
            logger.info(f"成功更新向量记忆: {len(memories)}")
            return True
            
        except Exception as e:
            logger.error(f"更新向量记忆失败: {e}")
            return False
    
    async def delete(self, memory_ids: list[str]) -> bool:
        """删除向量记忆"""
        try:
            await self.client.delete(self.collection_name, f'memory_id in {memory_ids}')
            
            logger.info(f"成功删除向量记忆: {len(memory_ids)}")
            return True
            
        except Exception as e:
            logger.error(f"删除向量记忆失败: {e}")
            return False
    
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
        self.loaded_collections: dict[str, MilvusMemoryCollection] = {}
        
        # 连接数据库
        self.client = AsyncMilvusClient(
            uri=self.url, 
            user=user, 
            password=password, 
            db_name=self.db_name, 
        )
    
    async def create_vector_memory(
        self,
        collection_name: str,
        dimension: int = 1536,
        index_type: str = "IVF_FLAT",
        metric_type: str = "COSINE", 
        **kwargs,
    ) -> Optional[MilvusMemoryCollection]:
        """创建向量记忆管理器"""
        try:
            # 定义字段模式
            fields = [
                FieldSchema(name="memory_id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="env_id", dtype=DataType.INT64),
                FieldSchema(name="agent_id", dtype=DataType.INT64),
                FieldSchema(name="task_id", dtype=DataType.INT64),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            ]
            # 创建集合模式
            schema = CollectionSchema(fields=fields, description=f"Vector memory collection: {collection_name}")
            
            # 创建索引参数
            index_params = await self.client.prepare_index_params(
                field_name="embedding", 
            )
            index_params.add_index(
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
            memory = MilvusMemoryCollection(
                collection_name=collection_name, 
                client=self.client, 
                **kwargs,
            )
            
            # 添加到已加载的集合中
            self.loaded_collections[collection_name] = memory
            logger.info(f"成功创建向量记忆管理器: {collection_name}")
            return memory
                
        except Exception as e:
            logger.error(f"创建向量记忆管理器失败: {e}")
            return None
    
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
