import asyncio
import json
import pytest
import tempfile
import os
import sys
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

from myagents.core.memories.milvus import MilvusManager, MilvusMemoryCollection
from myagents.core.memories.schemas import BaseVectorMemoryItem


# 全局fixture
@pytest.fixture
def temp_db_path():
    """创建临时数据库路径"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # 清理临时文件
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestMilvusMemoryCollection:
    """测试MilvusMemoryCollection类"""
    
    @pytest.fixture
    def mock_client(self):
        """创建模拟的Milvus客户端"""
        client = AsyncMock()
        return client
    
    @pytest.fixture
    def memory_collection(self, mock_client):
        """创建记忆集合实例"""
        return MilvusMemoryCollection(
            client=mock_client,
            collection_name="test_collection"
        )
    
    @pytest.fixture
    def sample_memories(self):
        """创建示例记忆数据"""
        return [
            BaseVectorMemoryItem(
                memory_id=1,
                env_id=1,
                agent_id=1,
                task_id=1,
                content="这是第一条测试记忆",
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5] * 300,  # 1500维向量
                metadata={"type": "test", "timestamp": "2024-01-01"}
            ),
            BaseVectorMemoryItem(
                memory_id=2,
                env_id=1,
                agent_id=1,
                task_id=1,
                content="这是第二条测试记忆",
                embedding=[0.2, 0.3, 0.4, 0.5, 0.6] * 300,  # 1500维向量
                metadata={"type": "test", "timestamp": "2024-01-02"}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_add_memories_success(self, memory_collection, sample_memories, mock_client):
        """测试成功添加记忆"""
        # 设置模拟返回值
        mock_client.insert.return_value = None
        
        # 执行测试
        result = await memory_collection.add(sample_memories)
        
        # 验证结果
        assert result is True
        mock_client.insert.assert_called_once_with(
            "test_collection",
            [memory.model_dump() for memory in sample_memories]
        )
    
    @pytest.mark.asyncio
    async def test_add_memories_failure(self, memory_collection, sample_memories, mock_client):
        """测试添加记忆失败"""
        # 设置模拟异常
        mock_client.insert.side_effect = Exception("插入失败")
        
        # 执行测试
        result = await memory_collection.add(sample_memories)
        
        # 验证结果
        assert result is False
    
    @pytest.mark.asyncio
    async def test_search_memories_success(self, memory_collection, mock_client):
        """测试成功搜索记忆"""
        # 准备查询向量
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 300
        
        # 模拟搜索结果
        mock_hit = MagicMock()
        mock_hit.score = 0.85
        mock_hit.get.side_effect = lambda key: {
            "memory_id": 1,
            "env_id": 1,
            "agent_id": 1,
            "task_id": 1,
            "content": "测试记忆内容",
            "embedding": query_embedding,
            "metadata": json.dumps({"type": "test"})
        }.get(key)
        
        mock_results = [[mock_hit]]
        mock_client.search.return_value = mock_results
        
        # 执行测试
        results = await memory_collection.search(
            env_id=1,
            agent_id=1,
            task_id=1,
            query_embedding=query_embedding,
            top_k=10,
            score_threshold=0.5
        )
        
        # 验证结果
        assert len(results) == 1
        memory, score = results[0]
        assert isinstance(memory, BaseVectorMemoryItem)
        assert score == 0.85
        assert memory.content == "测试记忆内容"
        
        # 验证搜索调用
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert call_args[1]["collection_name"] == "test_collection"
        assert call_args[1]["data"] == [query_embedding]
        assert call_args[1]["anns_field"] == "embedding"
        assert call_args[1]["limit"] == 10
    
    @pytest.mark.asyncio
    async def test_search_memories_below_threshold(self, memory_collection, mock_client):
        """测试搜索记忆分数低于阈值"""
        # 准备查询向量
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 300
        
        # 模拟搜索结果（分数低于阈值）
        mock_hit = MagicMock()
        mock_hit.score = 0.3  # 低于0.5阈值
        mock_results = [[mock_hit]]
        mock_client.search.return_value = mock_results
        
        # 执行测试
        results = await memory_collection.search(
            env_id=1,
            agent_id=1,
            task_id=1,
            query_embedding=query_embedding,
            top_k=10,
            score_threshold=0.5
        )
        
        # 验证结果
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_search_memories_failure(self, memory_collection, mock_client):
        """测试搜索记忆失败"""
        # 设置模拟异常
        mock_client.search.side_effect = Exception("搜索失败")
        
        # 执行测试
        results = await memory_collection.search(
            env_id=1,
            agent_id=1,
            task_id=1,
            query_embedding=[0.1] * 1500,
            top_k=10
        )
        
        # 验证结果
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_update_memories_success(self, memory_collection, sample_memories, mock_client):
        """测试成功更新记忆"""
        # 设置模拟返回值
        mock_client.delete.return_value = None
        mock_client.insert.return_value = None
        
        # 执行测试
        result = await memory_collection.update(sample_memories)
        
        # 验证结果
        assert result is True
        mock_client.delete.assert_called_once_with(
            "test_collection",
            "memory_id in [1, 2]"
        )
        mock_client.insert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_memories_failure(self, memory_collection, sample_memories, mock_client):
        """测试更新记忆失败"""
        # 设置模拟异常
        mock_client.delete.side_effect = Exception("删除失败")
        
        # 执行测试
        result = await memory_collection.update(sample_memories)
        
        # 验证结果
        assert result is False
    
    @pytest.mark.asyncio
    async def test_delete_memories_success(self, memory_collection, mock_client):
        """测试成功删除记忆"""
        memory_ids = ["1", "2", "3"]
        
        # 设置模拟返回值
        mock_client.delete.return_value = None
        
        # 执行测试
        result = await memory_collection.delete(memory_ids)
        
        # 验证结果
        assert result is True
        mock_client.delete.assert_called_once_with(
            "test_collection",
            "memory_id in ['1', '2', '3']"
        )
    
    @pytest.mark.asyncio
    async def test_delete_memories_failure(self, memory_collection, mock_client):
        """测试删除记忆失败"""
        memory_ids = ["1", "2"]
        
        # 设置模拟异常
        mock_client.delete.side_effect = Exception("删除失败")
        
        # 执行测试
        result = await memory_collection.delete(memory_ids)
        
        # 验证结果
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_collection_stats_success(self, memory_collection, mock_client):
        """测试成功获取集合统计信息"""
        # 模拟统计信息
        mock_stats = {
            "name": "test_collection",
            "description": "Test collection",
            "fields": []
        }
        mock_client.describe_collection.return_value = mock_stats
        
        # 执行测试
        stats = await memory_collection.get_collection_stats()
        
        # 验证结果
        assert stats == mock_stats
        mock_client.describe_collection.assert_called_once_with("test_collection")
    
    @pytest.mark.asyncio
    async def test_get_collection_stats_failure(self, memory_collection, mock_client):
        """测试获取集合统计信息失败"""
        # 设置模拟异常
        mock_client.describe_collection.side_effect = Exception("获取统计信息失败")
        
        # 执行测试
        stats = await memory_collection.get_collection_stats()
        
        # 验证结果
        assert stats == {}


class TestMilvusManager:
    """测试MilvusManager类"""
    
    @pytest.mark.asyncio
    async def test_init_with_url(self):
        """测试使用URL初始化"""
        manager = MilvusManager(url="http://localhost:19530")
        assert manager.url == "http://localhost:19530"
        assert manager.host is None
        assert manager.port is None
    
    @pytest.mark.asyncio
    async def test_init_with_host_port(self):
        """测试使用host和port初始化"""
        manager = MilvusManager(host="localhost", port="19530")
        assert manager.url == "http://localhost:19530"
        assert manager.host == "localhost"
        assert manager.port == "19530"
    
    @pytest.mark.asyncio
    async def test_init_with_defaults(self):
        """测试使用默认值初始化（Milvus Lite）"""
        manager = MilvusManager()
        assert manager.url == "./milvus_demo.db"
        assert manager.host is None
        assert manager.port is None
    
    @pytest.mark.asyncio
    async def test_init_with_conflict(self):
        """测试URL和host/port冲突"""
        with pytest.raises(ValueError, match="url和host、port不能同时提供"):
            MilvusManager(url="http://localhost:19530", host="localhost")
    
    @pytest.mark.asyncio
    @patch('myagents.core.memories.milvus.AsyncMilvusClient')
    async def test_create_vector_memory_success(self, mock_client_class, temp_db_path):
        """测试成功创建向量记忆管理器"""
        # 设置模拟客户端
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.prepare_index_params.return_value = AsyncMock()
        mock_client.create_collection.return_value = None
        
        # 创建管理器
        manager = MilvusManager(url=temp_db_path)
        
        # 执行测试
        memory_collection = await manager.create_vector_memory(
            collection_name="test_collection",
            dimension=1536
        )
        
        # 验证结果
        assert memory_collection is not None
        assert isinstance(memory_collection, MilvusMemoryCollection)
        assert memory_collection.collection_name == "test_collection"
        assert "test_collection" in manager.loaded_collections
        
        # 验证创建集合调用
        mock_client.create_collection.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('myagents.core.memories.milvus.AsyncMilvusClient')
    async def test_create_vector_memory_failure(self, mock_client_class, temp_db_path):
        """测试创建向量记忆管理器失败"""
        # 设置模拟异常
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.create_collection.side_effect = Exception("创建集合失败")
        
        # 创建管理器
        manager = MilvusManager(url=temp_db_path)
        
        # 执行测试
        memory_collection = await manager.create_vector_memory(
            collection_name="test_collection"
        )
        
        # 验证结果
        assert memory_collection is None
    
    @pytest.mark.asyncio
    @patch('myagents.core.memories.milvus.AsyncMilvusClient')
    async def test_drop_vector_memory_success(self, mock_client_class, temp_db_path):
        """测试成功删除向量记忆管理器"""
        # 设置模拟客户端
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.has_collection.return_value = True
        mock_client.drop_collection.return_value = None
        
        # 创建管理器
        manager = MilvusManager(url=temp_db_path)
        
        # 先创建集合
        memory_collection = await manager.create_vector_memory("test_collection")
        manager.loaded_collections["test_collection"] = memory_collection
        
        # 执行测试
        result = await manager.drop_vector_memory("test_collection")
        
        # 验证结果
        assert result is True
        assert "test_collection" not in manager.loaded_collections
        mock_client.drop_collection.assert_called_once_with("test_collection")
    
    @pytest.mark.asyncio
    @patch('myagents.core.memories.milvus.AsyncMilvusClient')
    async def test_drop_vector_memory_not_exists(self, mock_client_class, temp_db_path):
        """测试删除不存在的向量记忆管理器"""
        # 设置模拟客户端
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.has_collection.return_value = False
        
        # 创建管理器
        manager = MilvusManager(url=temp_db_path)
        
        # 执行测试
        result = await manager.drop_vector_memory("nonexistent_collection")
        
        # 验证结果
        assert result is False
        mock_client.drop_collection.assert_not_called()
    
    @pytest.mark.asyncio
    @patch('myagents.core.memories.milvus.AsyncMilvusClient')
    async def test_drop_vector_memory_failure(self, mock_client_class, temp_db_path):
        """测试删除向量记忆管理器失败"""
        # 设置模拟客户端
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.has_collection.return_value = True
        mock_client.drop_collection.side_effect = Exception("删除集合失败")
        
        # 创建管理器
        manager = MilvusManager(url=temp_db_path)
        
        # 执行测试
        result = await manager.drop_vector_memory("test_collection")
        
        # 验证结果
        assert result is False


class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    @patch('myagents.core.memories.milvus.AsyncMilvusClient')
    async def test_full_workflow(self, mock_client_class, temp_db_path):
        """测试完整的记忆管理工作流程"""
        # 设置模拟客户端
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.prepare_index_params.return_value = AsyncMock()
        mock_client.create_collection.return_value = None
        mock_client.insert.return_value = None
        mock_client.has_collection.return_value = True
        mock_client.drop_collection.return_value = None
        
        # 创建管理器
        manager = MilvusManager(url=temp_db_path)
        
        # 1. 创建向量记忆管理器
        memory_collection = await manager.create_vector_memory("test_collection")
        assert memory_collection is not None
        
        # 2. 创建测试记忆
        memories = [
            BaseVectorMemoryItem(
                memory_id=1,
                env_id=1,
                agent_id=1,
                task_id=1,
                content="测试记忆1",
                embedding=[0.1] * 1536,
                metadata={"type": "test"}
            ),
            BaseVectorMemoryItem(
                memory_id=2,
                env_id=1,
                agent_id=1,
                task_id=1,
                content="测试记忆2",
                embedding=[0.2] * 1536,
                metadata={"type": "test"}
            )
        ]
        
        # 3. 添加记忆
        result = await memory_collection.add(memories)
        assert result is True
        
        # 4. 删除记忆管理器
        result = await manager.drop_vector_memory("test_collection")
        assert result is True


def generate_test_embeddings(dimension: int = 1536, count: int = 5) -> List[List[float]]:
    """生成测试用的向量嵌入"""
    return [np.random.rand(dimension).tolist() for _ in range(count)]


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"]) 
