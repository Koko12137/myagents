from typing import Callable, Awaitable, Union

from fastmcp.tools import Tool as FastMcpTool
from loguru import logger

from myagents.core.interface import ToolsCaller, Stateful
from myagents.core.messages import ToolCallRequest, ToolCallResult
from myagents.core.interface import CallStack


class ToolsMixin(ToolsCaller):
    """ToolsMixin is a mixin class for tools management.
    
    Attributes:
        tools (dict[str, FastMcpTool]):
            The tools of the mixin.
        call_stack (CallStack):
            The call stack of the mixin.
    """
    tools: dict[str, FastMcpTool]
    call_stack: CallStack
    
    def __init__(self, call_stack: CallStack, *args, **kwargs) -> None:
        """Initialize the ToolsMixin.

        Args:
            call_stack (CallStack):
                The call stack of the mixin.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        super().__init__(*args, **kwargs)
        
        # Initialize the tools
        self.tools = {}
        # Initialize the call stack
        self.call_stack = call_stack
        
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
        
        # Create a FastMcpTool instance
        tool_obj = FastMcpTool.from_function(tool, tags=tags)
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
        target: Stateful,
        **kwargs,
    ) -> ToolCallResult:
        """Call a tool to control the workflow.
        
        Args:
            tool_call (ToolCallRequest):
                The tool call request.
            target (Stateful):
                The target of the tool call.
            **kwargs:
                The additional keyword arguments for calling the tool.
                
        Returns:
            ToolCallResult:
                The tool call result.
        """
        # Check if the tool name is registered
        if tool_call.name not in self.tools:
            raise ValueError(f"Tool {tool_call.name} is not registered.")
        
        # Push the kwargs to the call stack
        self.call_stack.call_next(key_values={"target": target, "tool_call": tool_call, **kwargs})
        
        # Call the tool
        try:
            # Call the tool
            result = await self.tools[tool_call.name].run(tool_call.args)
            # Format the result
            result = ToolCallResult(**result.structured_content)
        except Exception as e:
            # Log the error
            logger.error(f"Error calling tool {tool_call.name}: {e}")
            # Create a new tool call result
            result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=True, 
                content=f"工具调用失败: {e}"
            )
            
        # Resume the call stack
        self.call_stack.return_prev()
        
        # Return the tool call result
        return result
