from typing import Optional

from pydantic import BaseModel, Field


class VectorMemoryConfig(BaseModel):
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
        default="./myagents.db", 
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
        default="myagents", 
    )
    port: Optional[int] = Field(
        description="向量记忆数据库端口", 
        default=5432, 
    )
    host: Optional[str] = Field(
        description="向量记忆数据库主机", 
        default=None, 
    )
