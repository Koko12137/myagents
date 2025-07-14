import asyncio
from threading import Thread

from fastmcp import FastMCP, Client as FastMCPClient
from fastapi import FastAPI, Query, Body
import uvicorn


class ProxyMCP:
    """ProxyMCP is the proxy mcp server. It is used to proxy the mcp server to the client.
    
    Attributes:
        config (dict):
            The config of the proxy mcp server. 
        mcp (FastMCP):
            The mcp server.
        app (FastAPI):
            The REST API for register the mcp server dynamically.
    """
    config: dict
    # The mcp server
    mcp: FastMCP
    # The REST API for register the mcp server dynamically
    app: FastAPI
    # Connected mcp servers
    connected_mcp_servers: dict[str, FastMCPClient]
    
    def __init__(self, config: dict = {}, *args, **kwargs) -> None:
        """
        Initialize the ProxyMCP.
        
        Args:
            config (dict):
                The config of the proxy mcp server.
        """
        # Initialize the config
        self.config = config
        self.mcp = FastMCP(
            name=self.config.get("name", "ProxyMCP"),
            instructions=self.config.get("instructions", "ProxyMCP is the proxy mcp server. It is used to proxy the mcp server to the client."),
        )
        # Initialize the app
        self.app = FastAPI(
            title=self.config.get("title", "ProxyMCP"),
            description=self.config.get("description", "ProxyMCP is the proxy mcp server. It is used to proxy the mcp server to the client."),
            version=self.config.get("version", "0.1.0"),
            contact=self.config.get("contact", {}),
            license=self.config.get("license", {}),
            terms_of_service=self.config.get("terms_of_service", ""),
            openapi_url=self.config.get("openapi_url", "/openapi.json"),
        )
        self.connected_mcp_servers = {}
        
        # Post init
        self.post_init()
        
    def post_init(self) -> None:
        
        @self.app.post("/remote/register")
        async def register(
            url: str = Body(..., description="The URL of the mcp server."), 
            prefix: str = Body(default="", description="The prefix of the mcp server.")
        ) -> dict:
            # Check if the mcp server is already connected
            if prefix in self.connected_mcp_servers:
                return {"message": "MCP server already registered."}
            
            # Create a new client
            client = FastMCPClient(url)
            # Add the client to the connected mcp servers
            self.connected_mcp_servers[prefix] = client
            # Set as a proxy and mount
            proxy = FastMCP.as_proxy(client)
            # Mount the proxy to the mcp server
            self.mcp.mount(proxy, prefix=prefix)
            # Return the result
            return {"message": "MCP server registered successfully."}
        
        @self.app.post("/local/register")
        async def register(
            start_command: str = Body(description="The start command of the mcp server."), 
            args: list[str] = Body(description="The arguments to be passed to the mcp server."), 
            env: dict[str, str] = Body(description="The environment variables to be passed to the mcp server."), 
            prefix: str = Body(description="The prefix of the mcp server.")
        ) -> dict:
            from fastmcp.client.transports import StdioTransport

            transport = StdioTransport(
                command=start_command,
                args=args,
                env=env,
            )
            client = FastMCPClient(transport)
            # Add the client to the connected mcp servers
            self.connected_mcp_servers[prefix] = client
            # Set as a proxy and mount
            proxy = FastMCP.as_proxy(client)
            # Mount the proxy to the mcp server
            self.mcp.mount(proxy, prefix=prefix)
            # Return the result
            return {"message": "MCP server registered successfully."}
        
        @self.app.post("/remote/unregister")
        async def unregister(
            prefix: str = Body(description="The prefix of the mcp server.")
        ) -> dict:
            # Check if the mcp server is connected
            if prefix not in self.connected_mcp_servers:
                return {"message": "MCP server not registered."}
            
            # Unregister the mcp server
            self.mcp.unregister(prefix)
            # Remove the client from the connected mcp servers
            del self.connected_mcp_servers[prefix]
            # Return the result
            return {"message": "MCP server unregistered successfully."}
        
        @self.app.post("/local/unregister")
        async def unregister(
            prefix: str = Body(description="The prefix of the mcp server.")
        ) -> dict:
            # Check if the mcp server is connected
            if prefix not in self.connected_mcp_servers:
                return {"message": "MCP server not registered."}
            
            # Unregister the mcp server
            self.mcp.unregister(prefix, local=True)
            # Remove the client from the connected mcp servers
            del self.connected_mcp_servers[prefix]
            # Return the result
            return {"message": "MCP server unregistered successfully."}
        
    def run_restful(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        # Run the app
        uvicorn.run(self.app, host=host, port=port)
        
    def run_mcp(self, host: str = "0.0.0.0", port: int = 8001) -> None:
        # Run the mcp server
        self.mcp.run(transport="streamable-http", host=host, port=port)
    
    def run(self, host: str = "0.0.0.0", rest_port: int = 8000, mcp_port: int = 8001) -> None:
        # Create a new thread for running the restful server
        rest_thread = Thread(target=self.run_restful, args=(host, rest_port))
        # Create a new thread for running the mcp server
        mcp_thread = Thread(target=self.run_mcp, args=(host, mcp_port))
        # Start the threads
        rest_thread.start()
        mcp_thread.start()
        # Wait for the threads to finish
        rest_thread.join()
        mcp_thread.join()
