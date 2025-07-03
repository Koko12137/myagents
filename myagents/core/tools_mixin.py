from typing import Callable, Awaitable

from fastmcp.tools import Tool as FastMCPTool

from myagents.core.interface import Context
from myagents.core.message import ToolCallRequest, ToolCallResult


class ToolsMixin:
    """ToolsMixin is a mixin class for tools management.
    
    Attributes:
        tools (dict[str, FastMCPTool]):
            The tools of the mixin.
        context (Context):
            The context of the mixin.
    """
    tools: dict[str, FastMCPTool]
    context: Context
    
    def add_tool(
        self, 
        name: str, 
        tool: Callable[..., Awaitable[ToolCallResult]], 
        tags: list[str] = [],
    ) -> None:
        """Add a tool to the mixin.
        
        Args:
            name (str):
                The name of the tool.
            tool (Callable[..., Awaitable[ToolCallResult]]):
                The tool to add. This tool should return the tool call result. 
            tags (list[str], optional):
                The tags of the tool.
                
        Raises:
            ValueError:
                If the tool name is already registered.
        """
        # Check if the tool name is already registered
        if name in self.tools:
            raise ValueError(f"Tool {name} is already registered.")
        
        # Create a FastMCPTool instance
        tool_obj = FastMCPTool.from_function(tool, tags=tags)
        # Register the tool to the mixin
        self.tools[name] = tool_obj
        
    def register_tool(
        self, 
        name: str, 
        tags: list[str] = [], 
    ) -> Callable[..., Awaitable[ToolCallResult]]:
        """This is a FastAPI like decorator to register a tool to the mixin.
        
        Args:
            name (str):
                The name of the tool.
            tags (list[str], optional):
                The tags of the tool.
                
        Returns:
            Callable[..., Awaitable[ToolCallResult]]:
                The function registered.
        """
        # Define a wrapper function to register the tool
        def wrapper(
            func: Callable[..., Awaitable[ToolCallResult]]
        ) -> Callable[..., Awaitable[ToolCallResult]]:
            """Wrapper function to call the tool.
            
            Args:
                func (Callable[..., Awaitable[ToolCallResult]]):
                    The function to register.

            Returns:
                Callable[..., Awaitable[ToolCallResult]]:
                    The function registered.
            """
            self.add_tool(name, func, tags=tags)
            return func
        
        # Return the wrapper function
        return wrapper

    async def call_tool(
        self, 
        tool_call: ToolCallRequest, 
        **kwargs,
    ) -> ToolCallResult:
        """Call a tool to control the workflow.
        
        Args:
            tool_call (ToolCallRequest):
                The tool call request.
            **kwargs:
                The additional keyword arguments for calling the tool.
                
        Returns:
            ToolCallResult:
                The tool call result.
        """
        # Check if the tool name is registered
        if tool_call.name not in self.tools:
            raise ValueError(f"Tool {tool_call.name} is not registered.")
        
        # Put the keyword arguments to the context
        self.context.create_next(**kwargs)
        
        # Call the tool
        tool: Callable[..., Awaitable[ToolCallResult]] = self.tools[tool_call.name].fn
        # Call the tool
        result: ToolCallResult = await tool(tool_call.args)
        # Resume the context
        self.context.done()
        # Return the tool call result
        return result
