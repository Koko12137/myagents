from typing import Optional

from pydantic import BaseModel, Field


class VectorDBConfig(BaseModel):
    """向量记忆数据库的配置

    参数:
        type (str):
            向量记忆数据库类型
        url (str):
            向量记忆数据库的 URL
        username (str, 可选):
            向量记忆数据库用户名
        password (str, 可选):
            向量记忆数据库密码
        database (str, 可选):
            向量记忆数据库名称
        port (int, 可选):
            向量记忆数据库端口
        host (str, 可选):
            向量记忆数据库主机
    """
    type: str = Field(
        description="向量记忆数据库类型", 
        default="milvus", 
    )
    url: str = Field(
        description="向量记忆数据库 URL", 
        default=None, 
    )
    username: Optional[str] = Field(
        description="向量记忆数据库用户名", 
        default=None, 
    )
    password: Optional[str] = Field(
        description="向量记忆数据库密码", 
        default=None, 
    )
    database: Optional[str] = Field(
        description="向量记忆数据库名称", 
        default="myagents_default", 
    )
    port: Optional[int] = Field(
        description="向量记忆数据库端口", 
        default=None, 
    )
    host: Optional[str] = Field(
        description="向量记忆数据库主机", 
        default=None, 
    )
    
    
class VectorCollectionConfig(BaseModel):
    """向量记忆集合的配置
    
    参数:
        collection_name: 集合名称
        dimension: 集合维度
        metric_type: 集合度量类型，可选值为: COSINE, EUCLIDEAN, DOT_PRODUCT
        index_type: 集合索引类型，可选值为: IVF_FLAT, IVF_SQ8, IVF_PQ, HNSW, ANNOY, DISKANN
    """
    collection_name: str = Field(
        description="向量记忆集合名称", 
        default=None, 
    )
    dimension: Optional[int] = Field(
        description="向量记忆集合维度", 
        default=None, 
    )
    metric_type: Optional[str] = Field(
        description="向量记忆集合度量类型，可选值为: COSINE, EUCLIDEAN, DOT_PRODUCT", 
        default="COSINE", 
    )
    index_type: Optional[str] = Field(
        description="向量记忆集合索引类型，可选值为: IVF_FLAT, IVF_SQ8, IVF_PQ, HNSW, ANNOY, DISKANN", 
        default=None, 
    )
    vector_db: Optional[VectorDBConfig] = Field(
        description="向量记忆数据库的配置", 
        default=None, 
    )
    
