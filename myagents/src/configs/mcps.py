from pydantic import BaseModel, Field


class MCPConfig(BaseModel):
    server_name: str = Field(description="The name of the MCP server.")
    server_url: str = Field(description="The URL of the MCP server.")
    server_port: int = Field(description="The port of the MCP server.")
    auth_token: str = Field(description="The authentication token to use.")