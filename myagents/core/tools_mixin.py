from typing import Callable, Awaitable, Union

from fastmcp.tools import Tool as FastMCPTool
from loguru import logger

from myagents.core.interface import Context, ToolsCaller
from myagents.core.messages import ToolCallRequest, ToolCallResult


class ToolsMixin(ToolsCaller):
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
        tool: Union[Awaitable[ToolCallResult], Callable[..., ToolCallResult]], 
        tags: list[str] = [], 
        replace: bool = True,
    ) -> None:
        """Add a tool to the mixin. 
        
        Args:
            name (str):
                The name of the tool.
            tool (Union[Awaitable[ToolCallResult], Callable[..., ToolCallResult]]):
                The tool to add. This tool should return the tool call result. 
            tags (list[str], optional):
                The tags of the tool.
            replace (bool, optional, defaults to True):
                Whether to replace the tool if it is already registered. 
        
        Raises:
            ValueError:
                If the tool name is already registered and the replace flag is False.
        """
        # Check if the tool name is already registered
        if name in self.tools and not replace:
            raise ValueError(f"Tool {name} is already registered.")
        
        # Create a FastMCPTool instance
        tool_obj = FastMCPTool.from_function(tool, tags=tags)
        # Register the tool to the mixin
        self.tools[name] = tool_obj
        
    def register_tool(
        self, 
        name: str, 
        tags: list[str] = [], 
        replace: bool = True,
    ) -> Callable[..., Union[Awaitable[ToolCallResult], Callable[..., ToolCallResult]]]:
        """This is a FastAPI like decorator to register a tool to the mixin.
        
        Args:
            name (str):
                The name of the tool.
            tags (list[str], optional):
                The tags of the tool.
            replace (bool, optional, defaults to True):
                Whether to replace the tool if it is already registered.
                
        Returns:
            Callable[..., Union[Awaitable[ToolCallResult], Callable[..., ToolCallResult]]]:
                The tool function.
        """
        # Define a wrapper function to register the tool
        def wrapper(
            func: Union[Awaitable[ToolCallResult], Callable[..., ToolCallResult]]
        ) -> Union[Awaitable[ToolCallResult], Callable[..., ToolCallResult]]:
            """Wrapper function to call the tool.
            
            Args:
                func (Union[Awaitable[ToolCallResult], Callable[..., ToolCallResult]]):
                    The function to register.

            Returns:
                Union[Awaitable[ToolCallResult], Callable[..., ToolCallResult]]:
                    The tool function.
            """
            self.add_tool(name, func, tags=tags, replace=replace)
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
        
        # Put the tool_call and keyword arguments to the context
        self.context.create_next(tool_call=tool_call, **kwargs)
        
        # Get the tool
        tool = self.tools[tool_call.name].fn
        try:
            # Call the tool
            if isinstance(tool, Awaitable):
                result = await tool(tool_call.args)
            else:
                result = tool(tool_call.args)
        except Exception as e:
            # Log the error
            logger.error(f"Error calling tool {tool_call.name}: {e}")
            # Create a new tool call result
            result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=True, 
                content=f"工具调用失败: {e}"
            )
            
        # Resume the context
        self.context.done()
        # Return the tool call result
        return result
