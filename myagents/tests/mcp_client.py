import asyncio

from fastmcp import Client


async def main() -> None:
    # Initialize the client
    async with Client("http://localhost:8001/mcp") as client:
        # Ping the client
        print(await client.ping())
        # List the tools
        print(await client.list_tools())
    

if __name__ == "__main__":
    asyncio.run(main())
