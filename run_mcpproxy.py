import json

from myagents.mcps.proxy import ProxyMCP


if __name__ == "__main__":
    # Initialize the gate
    proxy_mcp = ProxyMCP()
    # Run the proxy mcp
    proxy_mcp.run()
