from mcp import Tool as MCPTool
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import Provider


DESCRIPTION_TYPE_MAP = {
    Provider.OPENAI: "parameters", 
}


def tool_schema(
    tool: MCPTool | FastMcpTool, 
    provider: Provider, 
) -> dict:
    """
    Get the schema of the tool. 
    
    Args:
        tool: MCPTool | FastMcpTool
            The tool to get the description of. 
        provider (Provider): 
            The provider of the tool. 
            
    Returns:
        dict 
            The schema of the tool.
            
    Raises:
        ValueError: If the provider is not supported. 
    """
    if provider not in DESCRIPTION_TYPE_MAP.keys():
        raise ValueError(f"Unsupported provider: {provider}")
    
    if provider == Provider.OPENAI:
        # This schema is only compatible with OpenAI. 
        description = {
            "type": "function", 
            "function": {
                "name": tool.name,
                "description": tool.description, 
                "strict": True, 
            }
        }

        if isinstance(tool, FastMcpTool):
            parameters = tool.parameters
            # parameters = parameters['properties']
            description['function']['annotations'] = tool.annotations
            description['function'][DESCRIPTION_TYPE_MAP[provider]] = parameters
        elif isinstance(tool, MCPTool):
            description['function'][DESCRIPTION_TYPE_MAP[provider]] = tool.inputSchema
            description['function']['annotations'] = tool.annotations
        else:
            raise ValueError(f"Unsupported tool type: {type(tool)}") 
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    return description


class ToolView:
    """This view is used to format the tool to a string.
    
    Attributes:
        tool (MCPTool | FastMcpTool):
            The tool to be viewed.
        template (str):
            The template of the tool view.
    """
    tool: MCPTool | FastMcpTool
    template: str = """
    ==== Function: {name} ====
    {description}
    """
    
    def __init__(self, tool: MCPTool | FastMcpTool) -> None:
        self.tool = tool
        
    def format(self) -> str:
        return self.template.format(
            name=self.tool.name,
            description=self.tool.description,
        )
