import asyncio
import os

import httpx


TAVILY_START_COMMAND = "npx"
TAVILY_ARGS = [
    "-y", 
    "tavily-mcp@0.1.3"
]
TAVILY_ENV = {
    "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
}


class TavilyLocalClient:
    
    name: str
    proxy_url: str
    client: httpx.AsyncClient
    start_command: str
    args: list[str]
    env: dict[str, str]
    
    def __init__(self, proxy_url: str, *args, **kwargs) -> None:
        """
        Initialize the TavilyLocalClient.
        
        Args:
            proxy_url (str):
                The URL of the proxy server.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        self.name = kwargs.get("name", "TavilyLocalClient")
        self.proxy_url = proxy_url
        self.client = httpx.AsyncClient()
        self.start_command = TAVILY_START_COMMAND
        self.args = TAVILY_ARGS
        self.env = TAVILY_ENV
        
    async def connect(self) -> None:
        # Call the proxy server to rigster the tavily local server
        response = await self.client.post(
            f"{self.proxy_url}/local/register",
            json={
                "start_command": self.start_command, 
                "args": self.args, 
                "env": self.env, 
                "prefix": self.name
            }
        )
        # Check the response
        if response.status_code != 200:
            raise Exception(f"Failed to register the tavily local server: {response.text}")
    
    async def disconnect(self) -> None:
        # Call the proxy server to unregister the tavily local server
        response = await self.client.post(
            f"{self.proxy_url}/local/unregister",
            json={
                "prefix": self.name
            }
        )
        # Check the response
        if response.status_code != 200:
            raise Exception(f"Failed to unregister the tavily local server: {response.text}")
    
    async def run(self) -> None:
        await self.connect()
        # Wait for the user to press Ctrl+C
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await self.disconnect()


async def main() -> None:
    # Initialize the tavily local client
    tavily_local_client = TavilyLocalClient(proxy_url="http://localhost:8000")
    # Run the tavily local client
    await tavily_local_client.run()


if __name__ == "__main__":
    asyncio.run(main())
