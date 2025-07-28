from pydantic import BaseModel, Field


class MCPConfig(BaseModel):
    """MCP 客户端的配置类
    
    属性:
        server_name (str):
            MCP 服务器的名称
        server_url (str):
            MCP 服务器的URL
        server_port (int):
            MCP 服务器的端口
        auth_token (str):
            要使用的认证令牌
    """
    server_name: str = Field(description="MCP 服务器的名称")
    server_url: str = Field(description="MCP 服务器的URL")
    server_port: int = Field(description="MCP 服务器的端口")
    auth_token: str = Field(description="要使用的认证令牌")
