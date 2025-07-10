from typing import Callable

from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import Agent, Stateful
from myagents.core.workflows.base import BaseWorkflow
from myagents.core.messages import SystemMessage, UserMessage, StopReason
from myagents.core.utils.context import BaseContext
from myagents.core.utils.extractor import extract_by_label
from myagents.prompts.workflows.react import PROFILE, SYSTEM_PROMPT, THINK_PROMPT, REFLECT_PROMPT


class ReActFlow(BaseWorkflow):
    """Reason and Act Flow is the workflow for the react agent.
    
    Attributes:
        profile (str):
            The profile of the workflow.
        agent (Agent):
            The agent that is used to reason and act. 
        prompts (dict[str, str]):
            The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
        context (BaseContext):
            The context of the workflow.
        tools (dict[str, FastMcpTool]):
            The tools of the workflow.
    """
    # Basic information
    profile: str
    agent: Agent
    prompts: dict[str, str]
    # Context and tools
    context: BaseContext
    tools: dict[str, FastMcpTool]
    
    def __init__(
        self, 
        profile: str = "", 
        system_prompt: str = "", 
        think_prompt: str = "", 
        reflect_prompt: str = "", 
        *args, 
        **kwargs,
    ) -> None:
        """Initialize the ReActFlow.

        Args:
            profile (str, optional, defaults to ""):
                The profile of the workflow.
            system_prompt (str, optional, defaults to ""):
                The system prompt of the workflow.
            think_prompt (str, optional, defaults to ""):
                The think prompt of the workflow.
            reflect_prompt (str, optional, defaults to ""):
                The reflect prompt of the workflow.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        super().__init__(*args, **kwargs)
        
        # Initialize the workflow components
        self.profile = profile if profile != "" else PROFILE
        self.agent = None
        # Update the prompts
        self.prompts = {
            "react_system": system_prompt if system_prompt != "" else SYSTEM_PROMPT.format(profile=self.profile),
            "react_think": think_prompt if think_prompt != "" else THINK_PROMPT,
            "react_reflect": reflect_prompt if reflect_prompt != "" else REFLECT_PROMPT,
        }
    
    async def run(
        self, 
        target: Stateful, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [], 
        running_checker: Callable[[Stateful], bool] = None, 
        *args, 
        **kwargs,
    ) -> Stateful:
        """Run the agent on the target. Before running the agent, you should get the lock of the agent. 
        
        Args:
            target (Stateful):
                The target to run the agent on.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of times to idle thinking the agent.
            tool_choice (str, optional, defaults to None):
                The designated tool choice to use for the agent. 
            exclude_tools (list[str], optional, defaults to []):
                The tools to exclude from the tool choice. 
            running_checker (Callable[[Stateful], bool], optional, defaults to None):
                The checker to check if the workflow should be running.
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            Stateful:
                The target after working with the workflow.
        """
        # Check if the running checker is provided
        if running_checker is None:
            # Set the running checker to the default checker
            running_checker = lambda target: target.is_running()
        
        # Run the workflow
        if running_checker(target):
            # Reason and act on the target
            target = await self.reason_act_reflect(
                target, 
                max_error_retry, 
                max_idle_thinking, 
                tool_choice, 
                exclude_tools, 
                running_checker, 
                *args, 
                **kwargs,
            )
        else:
            # Log the error
            logger.error("The target is not running, the workflow is not executed.")
            # Set the target to error
            target.to_error()
            
        # Return the target
        return target

    async def reason_act_reflect(
        self, 
        target: Stateful, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [], 
        running_checker: Callable[[Stateful], bool] = None, 
        *args, 
        **kwargs,
    ) -> Stateful:
        """Reason and act on the target.
        
        Args:
            target (Stateful):
                The target to reason and act on.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of times to idle thinking the agent. 
            tool_choice (str, optional, defaults to None):
                The designated tool choice to use for the agent. 
            exclude_tools (list[str], optional, defaults to []):
                The tools to exclude from the tool choice. 
            running_checker (Callable[[Stateful], bool], optional, defaults to None):
                The checker to check if the workflow should be running.
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent. 
                The following keyword arguments are supported:
                - react_system (str):
                    The system prompt of the workflow.
                - react_think (str):
                    The think prompt of the workflow.
                - react_reflect (str):
                    The reflect prompt of the workflow.
        """
        # Check if the running checker is provided
        if running_checker is None:
            # Set the running checker to the default checker
            running_checker = lambda target: target.is_running()
        
        # Prepare the prompts 
        react_system = kwargs.pop("react_system", self.prompts["react_system"])
        react_think = kwargs.pop("react_think", self.prompts["react_think"])
        react_reflect = kwargs.pop("react_reflect", self.prompts["react_reflect"])
        
        # Update system prompt to history
        message = SystemMessage(content=react_system)
        target.update(message)
        
        # Error and idle thinking control
        current_thinking = 0
        current_error = 0
        
        while running_checker(target):
        
            # === Reason Stage ===
            target, current_error, current_thinking = await self.reason_act(
                target, 
                react_think=react_think,
                max_error_retry=max_error_retry, 
                current_error=current_error, 
                max_idle_thinking=max_idle_thinking, 
                current_thinking=current_thinking, 
                tool_choice=tool_choice, 
                exclude_tools=exclude_tools,
                *args, 
                **kwargs,
            )
            # Check if the target is errored
            if target.is_error():
                # Force the loop to break
                break
            
            # === Reflect Stage ===
            target, finish_flag = await self.reflect(
                target, 
                react_reflect=react_reflect,
                *args, 
                **kwargs,
            )
            # Check if the target is finished
            if finish_flag:
                # Force the loop to break
                break
            
        return target
    
    async def reason_act(
        self, 
        target: Stateful, 
        react_think: str, 
        max_error_retry: int = 3, 
        current_error: int = 0, 
        max_idle_thinking: int = 1, 
        current_thinking: int = 0, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [], 
        *args, 
        **kwargs,
    ) -> tuple[Stateful, int, int]:
        """Reason and act on the target.
        
        Args:
            target (Stateful):
                The target to reason and act on.
            react_think (str):
                The think prompt of the workflow.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of times to retry the agent when the target is errored.
            current_error (int, optional, defaults to 0):
                The current error counter.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of times to idle thinking the agent. 
            current_thinking (int, optional, defaults to 0):
                The current thinking counter.
            tool_choice (str, optional, defaults to None):
                The designated tool choice to use for the agent. 
            exclude_tools (list[str], optional, defaults to []):
                The tools to exclude from the tool choice. 
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            tuple[Stateful, int, int]:
                The target, the current error counter and the current thinking counter.
        """
        # Prepare external tools
        external_tools = {**self.agent.tools, **self.agent.env.tools}
        
        # Observe the target
        observe = await self.agent.observe(target)
        # Log the observe
        logger.info(f"Observe: \n{observe}")
        # Create new user message
        message = UserMessage(content=react_think.format(observe=observe))
        # Update the target with the user message
        target.update(message)
        # Prepare the thinking kwargs
        think_kwargs = self.prepare_thinking_kwargs(
            tools=external_tools, 
            tool_choice=tool_choice, 
            exclude_tools=exclude_tools, 
            *args, 
            **kwargs,
        )
        # Think about the target
        message = await self.agent.think(target.get_history(), **think_kwargs)
        # Log the assistant message
        logger.info(f"Assistant Message: \n{message}")
        # Update the target with the assistant message
        target.update(message)
        
        # === Act Stage ===
        # Get all the tool calls from the assistant message
        if message.stop_reason == StopReason.TOOL_CALL:
            # Act on the task or environment
            for tool_call in message.tool_calls:
                # Reset the idle thinking counter
                current_thinking = 0
                # Act on the target
                result = await self.agent.act(
                    tool_call=tool_call, 
                    target=target, 
                    **kwargs, 
                )
                # Log the tool call result
                logger.info(f"Tool Call Result: \n{result}")
                # Update the target with the tool call results
                target.update(result)
                # Check if the tool call is errored
                if result.is_error:
                    # Increment the error counter
                    current_error += 1
                    # Notify the error limit to Agent
                    message = UserMessage(content=f"错误次数限制: {current_error}/{max_error_retry}，请重新思考，达到最大限制后将会被强制终止工作流。")
                    target.update(message)
                    # Log the error message
                    logger.info(f"Error Message: \n{message}")
                    # Check if the error counter is greater than the max error retry
                    if current_error >= max_error_retry:
                        # Set the task status to error
                        target.to_error()
                        # Force the react loop to finish
                        break
        else:
            # Increment the idle thinking counter
            current_thinking += 1
            # Notify the idle thinking limit to Agent
            message = UserMessage(content=f"空闲思考次数限制: {current_thinking}/{max_idle_thinking}，请重新思考，达到最大限制后将会被强制终止工作流。")
            target.update(message)
            # Log the idle thinking message
            logger.info(f"Idle Thinking Message: \n{message}")
            # Check if the idle thinking counter is greater than the max idle thinking
            if current_thinking >= max_idle_thinking:
                # Set the task status to error
                target.to_error()
        
        return target, current_error, current_thinking

    async def reflect(
        self, 
        target: Stateful, 
        react_reflect: str, 
        *args, 
        **kwargs,
    ) -> tuple[Stateful, bool]:
        """Reflect on the target.
        
        Args:
            target (Stateful):
                The target to reflect on.
            react_reflect (str):
                The reflect prompt of the workflow.
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent.
                The following keyword arguments are supported:
                - react_system (str):
                    The system prompt of the workflow.
                - react_think (str):
                    The think prompt of the workflow.
        
        Returns:
            tuple[Stateful, bool]:
                The target and the finish flag.
        """
        # Observe the target after acting
        observe = await self.agent.observe(target)
        # Log the observe
        logger.info(f"Observe: \n{observe}")
        # Create new user message
        message = UserMessage(content=react_reflect.format(observe=observe))
        # Update the target with the user message
        target.update(message)
        # Reflect the action taken on the target
        message = await self.agent.think(target.get_history(), tools=self.tools)
        # Log the assistant message
        logger.info(f"Assistant Message: \n{message}")
        # Update the target with the assistant message
        target.update(message)
        
        # === Finish Stage ===
        # Check if the finish flag is set
        finish_flag = extract_by_label(message.content, "finish", "finish_flag", "finish_workflow")
        if finish_flag == "True":
            finish_flag = True
        else:
            finish_flag = False
        
        return target, finish_flag
    
    def prepare_thinking_kwargs(
        self, 
        tools: dict[str, FastMcpTool] = {}, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [], 
        *args, 
        **kwargs,
    ) -> dict:
        """Prepare the thinking kwargs.
        
        Args:
            tools (dict[str, FastMcpTool], optional, defaults to {}):
                The tools to use for the agent. 
            tool_choice (str, optional, defaults to None):
                The designated tool choice to use for the agent. 
            exclude_tools (list[str], optional, defaults to []):
                The tools to exclude from the tool choice.
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            dict:
                The thinking kwargs.
        """
        # Prepare the thinking kwargs
        arguments = {}
        
        # Exclude the tools
        tools = {tool_name: tool for tool_name, tool in tools.items() if tool_name not in exclude_tools}
        
        # Set the tool choice
        if tool_choice is not None:
            # Convert the tool choice to a FastMcpTool
            tool = tools.get(tool_choice, tool_choice)
            arguments["tool_choice"] = tool
            arguments["tools"] = [tool]
        else:
            arguments["tools"] = tools
        
        return arguments
