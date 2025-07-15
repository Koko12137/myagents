from enum import Enum
from typing import Callable, Any

from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import Agent, Stateful
from myagents.core.workflows.base import BaseWorkflow
from myagents.core.messages import SystemMessage, UserMessage, StopReason
from myagents.core.utils.context import BaseContext
from myagents.core.utils.extractor import extract_by_label
from myagents.prompts.workflows.react import PROFILE, SYSTEM_PROMPT


class ReActStage(Enum):
    """The stage of the react workflow.

    - REASON_ACT: The reason and act stage.
    - REFLECT: The reflect stage.
    """
    INIT = 0
    REASON_ACT = 1
    REFLECT = 2


class ReActFlow(BaseWorkflow):
    """Reason and Act Flow is the workflow for the react agent.
    
    Attributes:
        profile (str):
            The profile of the workflow.
        agent (Agent):
            The agent that is used to reason and act. 
        context (BaseContext):
            The context of the workflow.
        tools (dict[str, FastMcpTool]):
            The tools of the workflow.
        stage (Enum):
            The stage of the workflow.
    """
    # Basic information
    profile: str
    agent: Agent
    # Context and tools
    context: BaseContext
    tools: dict[str, FastMcpTool]
    # Workflow stage
    stage: Enum
    
    def __init__(
        self, 
        profile: str = PROFILE, 
        *args, 
        **kwargs,
    ) -> None:
        """Initialize the ReActFlow.

        Args:
            profile (str, optional):
                The profile of the workflow.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        super().__init__(*args, **kwargs)
        
        # Initialize the workflow components
        self.profile = profile
        self.agent = None

    async def run(
        self, 
        target: Stateful, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        completion_config: dict[str, Any] = {}, 
        running_checker: Callable[[Stateful], bool] = None, 
        valid_stages: dict[str, Enum] = {
            "init": ReActStage.INIT, 
            "reason_act": ReActStage.REASON_ACT, 
            "reflect": ReActStage.REFLECT, 
        }, 
        *args, 
        **kwargs,
    ) -> Stateful:
        """Run the agent on the target. Before running the agent, you should get the lock of the agent. 
        
        Args:
            target (Stateful):
                The target to run the agent on.
            max_error_retry (int, optional):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional):
                The maximum number of times to idle thinking the agent.
            completion_config (dict[str, Any], optional):
                The completion config of the workflow. The following completion config are supported:
                1. "tool_choice": The tool choice to use for the agent. 
                2. "exclude_tools": The tools to exclude from the tool choice. 
            running_checker (Callable[[Stateful], bool], optional):
                The checker to check if the workflow should be running.
            valid_stages (dict[str, Enum], optional):
                The valid stages of the workflow. The key is the stage name and the value is the stage enum. 
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
                target=target, 
                max_error_retry=max_error_retry, 
                max_idle_thinking=max_idle_thinking, 
                completion_config=completion_config, 
                running_checker=running_checker, 
                valid_stages=valid_stages, 
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
        completion_config: dict[str, Any] = {}, 
        running_checker: Callable[[Stateful], bool] = None, 
        valid_stages: dict[str, Enum] = {
            "init": ReActStage.INIT, 
            "reason_act": ReActStage.REASON_ACT, 
            "reflect": ReActStage.REFLECT, 
        }, 
        *args, 
        **kwargs,
    ) -> Stateful:
        """Reason and act on the target.
        
        Args:
            target (Stateful):
                The target to reason and act on.
            max_error_retry (int, optional):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional):
                The maximum number of times to idle thinking the agent. 
            completion_config (dict[str, Any], optional):
                The completion config of the workflow. The following completion config are supported:
                1. "tool_choice": The tool choice to use for the agent. 
                2. "exclude_tools": The tools to exclude from the tool choice. 
            running_checker (Callable[[Stateful], bool], optional):
                The checker to check if the workflow should be running.
            valid_stages (dict[str, Enum], optional):
                The valid stages of the workflow. The key is the stage name and the value is the stage enum. 
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent. 
        """
        # Get the system prompt from the agent
        system_prompt = self.agent.prompts[valid_stages["init"]]
        # Update system prompt to history
        message = SystemMessage(content=system_prompt)
        target.update(message)
        
        # Error and idle thinking control
        current_thinking = 0
        current_error = 0
        
        while running_checker(target):
        
            # === Reason Stage ===
            # Reason and act on the target
            target, error_flag, tool_call_flag = await self.reason_act(
                target=target, 
                completion_config=completion_config, 
                to_stage=valid_stages["reason_act"], 
                *args, 
                **kwargs,
            )
            
            # Check if the error flag is set
            if error_flag:
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
                    # Record the error as answer
                    target.answer += f"\n\n错误次数限制已达上限: {current_error}/{max_error_retry}，错误原因: {target.get_history()[-1].content}"
                    # Force the react loop to finish
                    break
            
            # === Reflect Stage ===
            # Reflect on the target
            target, finish_flag = await self.reflect(
                target=target, 
                to_stage=valid_stages["reflect"], 
                *args, 
                **kwargs,
            )
            # Check if the target is finished
            if finish_flag:
                # Force the loop to break
                break
            
            # Check if the tool call flag is not set
            elif not tool_call_flag:
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
                    # Record the error as answer
                    target.answer += f"\n连续思考次数限制已达上限: {current_thinking}/{max_idle_thinking}，进入错误状态。"
            
        return target
    
    async def reason_act(
        self, 
        target: Stateful, 
        to_stage: Enum = ReActStage.REASON_ACT, 
        completion_config: dict[str, Any] = {}, 
        *args, 
        **kwargs,
    ) -> tuple[Stateful, bool, bool]:
        """Reason and act on the target.
        
        Args:
            target (Stateful):
                The target to reason and act on. 
            to_stage (Enum, optional):
                The stage to reason and act on. 
            completion_config (dict[str, Any], optional):
                The completion config of the workflow. The following completion config are supported:
                - "tool_choice": The tool choice to use for the agent. 
                - "exclude_tools": The tools to exclude from the tool choice. 
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            tuple[Stateful, bool, bool]:
                The target, the error flag and the tool call flag.
        """
        # Update the stage
        self.stage = to_stage
        
        # Initialize the error and tool call flag
        error_flag = False
        tool_call_flag = False
        
        # Prepare external tools
        external_tools = {**self.agent.tools, **self.agent.env.tools}
        
        # Observe the target
        observe = await self.agent.observe(target)
        # Log the observe
        logger.info(f"Observe: \n{observe}")
        # Create new user message with the think prompt
        message = UserMessage(content=observe)
        # Update the target with the user message
        target.update(message)
        # Prepare the thinking kwargs
        think_kwargs = self.prepare_thinking_kwargs(
            tools=external_tools, 
            tool_choice=completion_config.get("tool_choice", None), 
            exclude_tools=completion_config.get("exclude_tools", []), 
            *args, 
            **kwargs,
        )
        # Think about the target
        message = await self.agent.think(target.get_history(), **think_kwargs)
        # Log the assistant message
        if logger.level == "DEBUG":
            logger.debug(f"{str(self.agent)}: \n{message}")
        else:
            logger.info(f"{str(self.agent)}: \n{message.content}")
        # Update the target with the assistant message
        target.update(message)
        
        # === Act Stage ===
        # Get all the tool calls from the assistant message
        if message.stop_reason == StopReason.TOOL_CALL:
            # Set the tool call flag to True
            tool_call_flag = True
            
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
                    # Set the error flag to True
                    error_flag = True
        
        return target, error_flag, tool_call_flag

    async def reflect(
        self, 
        target: Stateful, 
        to_stage: Enum = ReActStage.REFLECT, 
        *args, 
        **kwargs,
    ) -> tuple[Stateful, bool]:
        """Reflect on the target.
        
        Args:
            target (Stateful):
                The target to reflect on.
            to_stage (Enum, optional):
                The stage to reflect on.
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent.
        
        Returns:
            tuple[Stateful, bool]:
                The target and the finish flag.
        """
        # Update the stage
        self.stage = to_stage
        
        # Observe the target after acting
        observe = await self.agent.observe(target)
        # Log the observe
        logger.info(f"Observe: \n{observe}")
        # Create new user message with the reflect prompt
        message = UserMessage(content=observe)
        # Update the target with the user message
        target.update(message)
        # Reflect the action taken on the target
        message = await self.agent.think(target.get_history(), tools=self.tools)
        # Log the assistant message
        if logger.level == "DEBUG":
            logger.debug(f"{str(self.agent)}: \n{message}")
        else:
            logger.info(f"{str(self.agent)}: \n{message.content}")
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
            tools (dict[str, FastMcpTool], optional):
                The tools to use for the agent. 
            tool_choice (str, optional):
                The designated tool choice to use for the agent. 
            exclude_tools (list[str], optional):
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
            # Check if the tool choice is in the tools
            assert tool_choice in tools, f"The tool choice {tool_choice} is not in the tools."
            # Set the tool choice
            arguments["tool_choice"] = tool_choice
            # Set the tools
            arguments["tools"] = [tools[tool_choice]]
        else:
            arguments["tools"] = tools
        
        return arguments
