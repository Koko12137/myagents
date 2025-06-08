from typing import Protocol, runtime_checkable
from contextlib import AsyncExitStack
from enum import Enum

import uvicorn
from loguru import logger
from pydantic import BaseModel, Field
from mcp import ClientSession, Tool
from mcp.client.stdio import stdio_client, StdioServerParameters
from fastapi import FastAPI, Request
from fastmcp.server import FastMCP

from myagents.src.interface import Environment, Logger
from myagents.src.utils.logger import init_logger
from myagents.prompts.envs.base import TOOL_CALL_PROMPT


class ToolResultStatus(Enum):
    """ToolResultStatus is the status of a tool call.
    
    Attributes:
        SUCCESS (str):
            The tool call is successful.
        FAILURE (str):
            The tool call is failed.
    """
    SUCCESS = "success"
    FAILURE = "failure"
    

class ToolResult(BaseModel, Environment):
    """ToolResult is the result of a tool call.
    
    Attributes:
        status (ToolResultStatus):
            The status of the tool call. 
        result (any): 
            The result of the tool call. 
    """
    status: ToolResultStatus = Field(description="The status of the tool call.")
    result: any = Field(description="The result of the tool call.")
    
    def observe(self) -> str:
        return TOOL_CALL_PROMPT.format(status=self.status, result=self.result)
    
    
class GateServerConfig(BaseModel):
    """GateServerConfig is the configuration for the Gate Server.
    
    Attributes:
        server_name: str
            The name of the MCP Gate Server.
        server_description: str
            The description of the MCP Gate Server.
        server_version: str
            The version of the MCP Gate Server.
        server_url: str
            The URL of the MCP Gate Server.
    """
    server_name: str = Field(description="The name of the MCP Gate Server.")
    server_description: str = Field(description="The description of the MCP Gate Server.")
    server_version: str = Field(description="The version of the MCP Gate Server.")
    server_url: str = Field(description="The URL of the MCP Gate Server.")
    port: int = Field(description="The port of the MCP Gate Server.")
    
    
class MCPServerConfig(BaseModel):
    """MCPServerConfig is the configuration for the MCP Server.
    
    Attributes:
        servers: list[dict[str, any]] 
            The servers'name and the server's config of the MCP Server Initialization.
    """
    servers: list[dict[str, any]] = Field(description="The servers'name and the server's config of the MCP Server Initialization.")
    
    
@runtime_checkable
class MCPGate(Protocol):
    """MCPGate is a protocol for Model Context Protocol Server Gate. All the MCP Clients 
    should connect to this gate to get the tools and sessions.
    
    Attributes:
        server_config: GateServerConfig
            The configuration of the MCP Gate Server.
        mcp_server_config: MCPServerConfig
            The configuration of the MCP Server.
        debug: bool
            The debug flag of the MCP Gate Server.
        initialized: bool
            The initialized flag of the MCP Gate Server.
            
        sessions: dict[str, ClientSession]
            The sessions of the MCP Gate. 
        tools: dict[str, dict[str, Tool]]
            The tools of the MCP Gate. The first layer of the dictionary is the session name, and the second 
            layer of the dictionary is the tool name. 
        descriptions: dict[str, dict[str, str]]
            The descriptions of the MCP Gate. The first layer of the dictionary is the session name, and the 
            second layer of the dictionary is the tool name.
        exit_stack: AsyncExitStack
            The exit stack of the MCP Gate.
    """
    server_config: GateServerConfig
    mcp_server_config: MCPServerConfig
    debug: bool
    initialized: bool
    
    sessions: dict[str, ClientSession]
    tools: dict[str, Tool]
    descriptions: dict[str, dict[str, str]]
    exit_stack: AsyncExitStack
    
    async def post_init_mcps(self) -> None:
        pass
    
    async def post_init_app(self) -> None:
        pass
    
    async def call(self, tool_name: str, args: dict[str, any]) -> any:
        pass
    
    async def connect(self, server_name: str, parameters: StdioServerParameters) -> None:
        pass


class BaseMCPGateServer:
    """BaseMCPGateServer is a protocol for Model Context Protocol Server Gate. All the MCP Clients should 
    connect to this gate to get the tools and sessions.
    
    Attributes:
        server_config: GateServerConfig
            The configuration of the MCP Gate Server.
        mcp_server_config: MCPServerConfig
            The configuration of the MCP Server.
        logger: Logger
            The logger of the MCP Gate Server.
        debug: bool
            The debug flag of the MCP Gate Server.
        initialized: bool
            The initialized flag of the MCP Gate Server.
            
        sessions: dict[str, ClientSession]
            The sessions of the MCP Gate.
        tools: dict[str, dict[str, Tool]]
            The tools of the MCP Gate. The first layer of the dictionary is the session name, and the second 
            layer of the dictionary is the tool name. 
        descriptions: dict[str, dict[str, str]]
            The descriptions of the MCP Gate. The first layer of the dictionary is the session name, and the 
            second layer of the dictionary is the tool name.
        exit_stack: AsyncExitStack
            The exit stack of the MCP Gate.
        app: FastAPI
            The fastapi app of the MCP Gate. 
    """
    server_config: GateServerConfig
    mcp_server_config: MCPServerConfig
    logger: any
    debug: bool
    initialized: bool
    
    sessions: dict[str, ClientSession]
    tools: dict[str, dict[str, Tool]]
    descriptions: dict[str, dict[str, str]]
    exit_stack: AsyncExitStack
    app: FastAPI
    
    def __init__(
        self, 
        server_config: GateServerConfig, 
        mcp_server_config: MCPServerConfig, 
        custom_logger: any = None, 
        debug: bool = False, 
    ) -> None:
        self.server_config = server_config
        self.mcp_server_config = mcp_server_config
        self.logger = custom_logger if custom_logger else logger
        self.debug = debug
        self.initialized = False
        
        self.sessions = {}
        self.tools = {}
        self.descriptions = {} # {session_name: {tool_name: tool_description}}
        self.exit_stack = AsyncExitStack()
        
        # Initialize the fastapi app
        app_kwargs = {
            "debug": debug, 
            "title": self.server_config.server_name, 
            "description": self.server_config.server_description, 
            "version": self.server_config.server_version, 
        }
        self.app = FastAPI(**app_kwargs)
        
    async def post_init_mcps(self) -> None:
        # Traverse the mcp_server_config and initialize the sessions and tools
        for server_name, server_config in self.mcp_server_config.servers:
            # Check if the server is already initialized
            if server_name in self.sessions:
                self.logger.debug(f"Server {server_name} already initialized")
                continue
            
            # Format the server config
            server_config = StdioServerParameters(**server_config)
            # Connect to the server
            await self.connect(server_config)
        
        tools = {}
        descriptions = {}
            
        # Initialize the tools
        for server_name, server in self.sessions.items():
            tools[server_name] = {}
            descriptions[server_name] = {}
            
            # Get the tools
            tool_list: list[Tool] = await server.list_tools()
            
            for tool in tool_list:
                # Add the tool to the tools dictionary
                tools[server_name][f"{server_name}::{tool.name}"] = tool
                
                # Add the tool description to the descriptions dictionary
                descriptions[server_name][f"{server_name}::{tool.name}"] = {
                    "type": "function", 
                    "function": {
                        "name": f"{server_name}::{tool.name}",
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                }
        
        # Update the tools and descriptions
        self.tools = tools
        self.descriptions = descriptions
        
        # Update the initialized flag
        self.initialized = True

    async def connect(self, server_name: str, parameters: StdioServerParameters) -> None:
        # Create stdio client
        stdio, writer = await self.exit_stack.enter_async_context(stdio_client(parameters))
        # Create the session
        session = await self.exit_stack.enter_async_context(ClientSession(stdio, writer))
        # Add the session to the sessions dictionary
        self.sessions[server_name] = session
    
    async def call(self, tool_name: str, args: dict[str, any]) -> any:
        # Split the server name and the tool name
        server_name, tool_name = tool_name.split("::")
        # Get the tool
        tool = self.tools[server_name][tool_name]
        # Call the tool
        return await tool.call(args)
    
    def post_init_app(self) -> None:
        
        @self.app.get("/tools")
        async def list_tools(request: Request) -> list[Tool]:
            self.logger.debug("Listing tools")
            tools = []
            for _, server_tools in self.tools.items():
                for tool_name, tool in server_tools.items():
                    tool.name = tool_name
                    tools.append(tool)
            return tools
        
        @self.app.post("/call")
        async def call_tool(
            tool_name: str, 
            args: dict[str, any], 
            request: Request, 
        ) -> ToolResult:
            try:
                self.logger.debug(f"Calling tool {tool_name} with args {args}")
                result = await self.call(tool_name, args)   
                return ToolResult(result=result, status=ToolResultStatus.SUCCESS)
            except Exception as e:
                self.logger.error(f"Error calling tool {tool_name}: {e}")
                return ToolResult(result=str(e), status=ToolResultStatus.FAILURE)
        
        
async def create_mcp_gate(
    server_config: GateServerConfig, 
    mcp_server_config: MCPServerConfig,     
    debug: bool = False, 
    log_file: str = "logs/mcp_gate.log", 
    rotation: str = "10 MB", 
    retention: str = "10 days", 
) -> MCPGate:
    # Initialize the logger
    init_logger(level="DEBUG" if debug else "INFO", sink=log_file, rotation=rotation, retention=retention)
    
    # Initialize the gate   
    gate = BaseMCPGateServer(server_config, mcp_server_config, debug)
    # Connect the gate to the MCP Server
    await gate.post_init_mcps()
    # Initialize the fastapi app
    gate.post_init_app()
    
    return gate


def run_mcp_gate(
    server_config: GateServerConfig, 
    mcp_server_config: MCPServerConfig,     
    debug: bool = False, 
    rotation: str = "10 MB", 
    retention: str = "10 days", 
) -> None:
    # Initialize the gate
    gate = create_mcp_gate(server_config, mcp_server_config, debug, rotation, retention)
    
    # Run the fastapi app
    uvicorn.run(
        app=gate.app, 
        host=server_config.server_url, 
        port=server_config.port, 
        proxy_headers=True, 
        forwarded_allow_ips="*", 
        reload=debug, 
    )
