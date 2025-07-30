import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from loguru import logger
from pymilvus import (
    AsyncMilvusClient, 
    Collection, 
    CollectionSchema, 
    FieldSchema, 
    DataType,
    SearchFuture,
)

from myagents.core.memories.schemas import BaseVectorMemory


class MilvusMemoryItem(BaseVectorMemory):
    """基于Milvus的向量记忆管理类"""
    
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class MilvusMemoryCollection:
    """基于Milvus的向量记忆管理类"""
    
    def __init__(
        self,
        collection_name: str,
        collection: Collection,
        dimension: int = 1536,
        index_type: str = "IVF_FLAT",
        metric_type: str = "COSINE",
        **kwargs, 
    ):
        """
        初始化向量记忆管理器
        
        Args:
            collection_name: 集合名称
            dimension: 向量维度
            index_type: 索引类型
            metric_type: 距离度量类型
            nlist: IVF索引的聚类数量
        """
        # 集合
        self.collection_name = collection_name
        self.collection = collection
        self.dimension = dimension
        self.index_type = index_type
        self.metric_type = metric_type
        if self.index_type == "IVF_FLAT":
            self.nlist = kwargs.get("nlist", 1024)
        elif self.index_type == "HNSW":
            self.m = kwargs.get("m", 16)
            self.ef_construction = kwargs.get("ef_construction", 100)
            self.ef_search = kwargs.get("ef_search", 10)
        elif self.index_type == "ANNOY":
            self.n_trees = kwargs.get("n_trees", 10)
            self.search_k = kwargs.get("search_k", 10)
            
    
    async def add(self, memories: list[MilvusMemoryItem]) -> bool:
        """插入向量记忆"""
        try:
            if not self.collection:
                self.collection = Collection(self.collection_name)
            
            # 准备数据
            data = [memory.model_dump() for memory in memories]
            
            # 插入数据
            self.collection.insert(data)
            self.collection.flush()
            
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
        score_threshold: float = 0.5
    ) -> list[tuple[MilvusMemoryItem, float]]:
        """搜索相似向量记忆"""
        try:
            if not self.collection:
                self.collection = Collection(self.collection_name)
            
            # 加载集合
            self.collection.load()
            
            # 查询条件
            expr = f"env_id == {env_id} and agent_id == {agent_id} and task_id == {task_id}"
            
            # 执行搜索
            results: SearchFuture = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                limit=top_k,
                output_fields=[
                    "env_id", "agent_id", "task_id", "memory_id", 
                    "content", "embedding", "metadata", 
                    "created_at", "updated_at"
                ],
                expr=expr,
                params={"nprobe": 10}, 
                _async=True,
            )
            
            # 解析结果
            memories = []
            for hits in results.result():
                for hit in hits:
                    if hit.score >= score_threshold:
                        memory = MilvusMemoryItem(
                            env_id=hit.entity.get("env_id"),
                            agent_id=hit.entity.get("agent_id"),
                            task_id=hit.entity.get("task_id"),
                            memory_id=hit.entity.get("memory_id"),
                            content=hit.entity.get("content"),
                            embedding=hit.entity.get("embedding"),
                            metadata=json.loads(hit.entity.get("metadata")),
                            created_at=datetime.fromisoformat(hit.entity.get("created_at")),
                            updated_at=datetime.fromisoformat(hit.entity.get("updated_at"))
                        )
                        memories.append((memory, hit.score))
            
            return memories
            
        except Exception as e:
            logger.error(f"搜索向量记忆失败: {e}")
            return []
    
    async def update(self, memories: list[MilvusMemoryItem]) -> bool:
        """更新向量记忆"""
        try:
            if not self.collection:
                self.collection = Collection(self.collection_name)
            
            # 删除旧记录
            memory_ids = [memory.memory_id for memory in memories]
            self.collection.delete(f'memory_id in {memory_ids}')
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
            if not self.collection:
                self.collection = Collection(self.collection_name)
            
            self.collection.delete(f'memory_id in {memory_ids}')
            self.collection.flush()
            
            logger.info(f"成功删除向量记忆: {len(memory_ids)}")
            return True
            
        except Exception as e:
            logger.error(f"删除向量记忆失败: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            if not self.collection:
                self.collection = Collection(self.collection_name)
            
            stats = {
                "name": self.collection_name,
                "num_entities": self.collection.num_entities,
                "schema": str(self.collection.schema),
                "indexes": self.collection.indexes
            }
            return stats
            
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
            elif host is None:
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
            # 检查集合是否已存在
            if self.client.has_collection(collection_name):
                logger.critical(f"集合已存在: {collection_name}")
                raise ValueError(f"集合已存在: {collection_name}")

            # 定义字段模式
            fields = [
                FieldSchema(name="env_id", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="agent_id", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="task_id", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="memory_id", dtype=DataType.VARCHAR, max_length=255, is_primary=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="updated_at", dtype=DataType.VARCHAR, max_length=50)
            ]
            # 创建集合模式
            schema = CollectionSchema(fields=fields, description=f"Vector memory collection: {self.collection_name}")
            
            # 创建索引参数
            index_params = await self.client.prepare_index_params(
                field_name="embedding", 
            )
            index_params.add_index(
                index_type=index_type, 
                metric_type=metric_type, 
            )
            
            # 创建集合
            collection = await self.client.create_collection(
                collection_name=collection_name, 
                dimension=dimension, 
                schema=schema, 
                index_params=index_params, 
            )
            
            # 创建新的向量记忆管理器
            memory = MilvusMemoryCollection(
                collection_name=collection_name, 
                collection=collection, 
                dimension=dimension, 
                index_type=index_type, 
                metric_type=metric_type, 
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
            if collection_name in self.loaded_collections:
                # 获取集合
                memory = self.loaded_collections[collection_name]
                # 卸载集合
                memory.collection.release()
                # 删除集合
                memory.collection.drop()
                del self.loaded_collections[collection_name]
                return True
            else:
                # 直接删除集合
                if self.client.has_collection(collection_name):
                    self.client.drop_collection(collection_name)
                    logger.info(f"成功删除向量记忆集合: {collection_name}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"删除向量记忆管理器失败: {e}")
            return False
    
    async def get_vector_memory(self, collection_name: str) -> Optional[MilvusMemoryCollection]:
        """获取向量记忆管理器"""
        try:
            if collection_name in self.loaded_collections:
                return self.loaded_collections[collection_name]
            
            # 如果集合存在，创建管理器
            if self.client.has_collection(collection_name):
                # 从数据库中获取集合
                collection = self.client.get_collection(collection_name)
                # 创建向量记忆管理器
                memory = MilvusMemoryCollection(
                    collection_name=collection_name, 
                    collection=collection, 
                    dimension=collection.schema.dimension, 
                    index_type=collection.index.index_type, 
                    metric_type=collection.index.metric_type, 
                )
                self.loaded_collections[collection_name] = memory
                return memory
            
            return None
            
        except Exception as e:
            logger.error(f"获取向量记忆管理器失败: {e}")
            return None
    
    async def list_collections(self) -> Dict[str, List[str]]:
        """列出所有集合"""
        try:
            collections = self.client.list_collections()
            vector_collections = []
            kv_collections = []
            
            for collection_name in collections:
                if collection_name in self.loaded_collections:
                    vector_collections.append(collection_name)
                elif collection_name in self.loaded_collections:
                    kv_collections.append(collection_name)
                else:
                    # 尝试判断集合类型
                    try:
                        collection = Collection(collection_name)
                        schema = collection.schema
                        # 检查是否有embedding字段来判断是否为向量集合
                        has_embedding = any(field.name == "embedding" for field in schema.fields)
                        if has_embedding:
                            vector_collections.append(collection_name)
                        else:
                            kv_collections.append(collection_name)
                    except:
                        kv_collections.append(collection_name)
            
            return {
                "vector_collections": vector_collections,
                "kv_collections": kv_collections
            }
            
        except Exception as e:
            logger.error(f"列出集合失败: {e}")
            return {"vector_collections": [], "kv_collections": []}
    
    async def get_server_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        try:
            info = {
                "host": self.host,
                "port": self.port,
                "collections": await self.list_collections(),
                "vector_memories": list(self.loaded_collections.keys()),
            }
            return info
        except Exception as e:
            logger.error(f"获取服务器信息失败: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            # 尝试列出集合来检查连接
            self.client.list_collections()
            return True
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False 
