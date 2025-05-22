import json

from myagents.src.mcps.gate import GateServerConfig, MCPServerConfig, run_mcp_gate


if __name__ == "__main__":
    # Load the gate config
    with open("configs/gate_config.json", "r") as f:
        gate_config = json.load(f)
        
    # Load the mcp config
    with open("configs/mcp_config.json", "r") as f:
        mcp_config = json.load(f)
        
    # Initialize the gate
    run_mcp_gate(
        GateServerConfig(**gate_config), 
        MCPServerConfig(**mcp_config), 
        debug=True, 
        rotation="10 MB", 
        retention="10 days", 
    )
