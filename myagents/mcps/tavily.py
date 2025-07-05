import httpx


TAVILY_START_COMMAND = "npx -y tavily-mcp@0.1.3"


class TavilyLocalClient:
    
    name: str
    proxy_url: str
    client: httpx.AsyncClient
    start_command: str
    
    def __init__(self, proxy_url: str, start_command: str = "", *args, **kwargs) -> None:
        self.name = kwargs.get("name", "TavilyLocalClient")
        self.proxy_url = proxy_url
        self.client = httpx.AsyncClient()
        self.start_command = start_command or TAVILY_START_COMMAND
        
    async def connect(self) -> None:
        # Call the proxy server to rigster the tavily local server
        response = await self.client.post(
            f"{self.proxy_url}/register/local",
            json={"start_command": self.start_command, "prefix": self.name}
        )
        # Check the response
        if response.status_code != 200:
            raise Exception(f"Failed to register the tavily local server: {response.text}")
    
    async def disconnect(self) -> None:
        # Call the proxy server to unregister the tavily local server
        response = await self.client.post(
            f"{self.proxy_url}/unregister/local",
            json={"start_command": self.start_command, "prefix": self.name}
        )
        # Check the response
        if response.status_code != 200:
            raise Exception(f"Failed to unregister the tavily local server: {response.text}")
    
    async def run(self) -> None:
        await self.connect()
