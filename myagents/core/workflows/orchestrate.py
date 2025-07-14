import json
from typing import Callable, Any

from json_repair import repair_json
from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import Agent, Stateful, Context, TreeTaskNode
from myagents.core.messages import AssistantMessage, UserMessage, SystemMessage, StopReason
from myagents.core.workflows.react import ReActFlow
from myagents.core.tasks import BaseTreeTaskNode, ToDoTaskView
from myagents.core.utils.extractor import extract_by_label
from myagents.core.utils.strings import normalize_string
from myagents.prompts.workflows.orchestrate import PROFILE, SYSTEM_PROMPT, THINK_PROMPT, ACTION_PROMPT, REFLECT_PROMPT


class OrchestrateFlow(ReActFlow):
    """This is use for Orchestrating the task. This workflow will not design any detailed plans, it will 
    only orchestrate the key objectives of the task. 
        
    Attributes:
        profile (str):
            The profile of the workflow.
        agent (Agent): 
            The agent that is used to orchestrate the task.
        prompts (dict[str, str]):
            The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
        context (Context):
            The global context container of the workflow.
        tools (dict[str, FastMcpTool]):
            The tools can be used for the agent. 
    """
    # Basic information
    profile: str
    agent: Agent
    prompts: dict[str, str]
    # Context and tools
    context: Context
    tools: dict[str, FastMcpTool]

    def __init__(
        self, 
        profile: str = "", 
        system_prompt: str = "", 
        think_prompt: str = "", 
        react_system: str = "", 
        react_think: str = "", 
        react_reflect: str = "", 
        *args, 
        **kwargs,
    ) -> None:
        """Initialize the OrchestrateFlow.

        Args:
            profile (str, optional, defaults to ""):
                The profile of the workflow.
            system_prompt (str, optional, defaults to ""):
                The system prompt of the workflow.
            think_prompt (str, optional, defaults to ""):
                The think prompt of the workflow.
            react_system (str, optional, defaults to ""):
                The react system prompt of the workflow.
            react_think (str, optional, defaults to ""):
                The action prompt of the workflow.
            react_reflect (str, optional, defaults to ""):
                The reflection prompt of the workflow.
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
        self.prompts.update({
            "orchestrate_system": system_prompt if system_prompt != "" else SYSTEM_PROMPT.format(profile=self.profile),
            "orchestrate_think": think_prompt if think_prompt != "" else THINK_PROMPT,
            "react_system": react_system if react_system != "" else SYSTEM_PROMPT.format(profile=self.profile),
            "react_think": react_think if react_think != "" else ACTION_PROMPT,
            "react_reflect": react_reflect if react_reflect != "" else REFLECT_PROMPT,
        })
    
    def create_task(
        self, 
        parent: TreeTaskNode, 
        orchestration: str, 
        current_error: int = 0, 
    ) -> tuple[UserMessage, int]:
        """Create a new task based on the orchestration blueprint.
        
        Args:
            parent (TreeTaskNode):
                The parent task to create the new task.
            orchestration (str):
                The orchestration blueprint to create the new task.
            current_error (int, optional, defaults to 0):
                The current error counter. 
                
        Returns:
            UserMessage:
                The user message after creating the new task.
            int:
                The current error counter.
        """
        try:
            # Repair the json
            orchestration = repair_json(orchestration)
            # Parse the orchestration
            orchestration: dict[str, dict[str, str]] = json.loads(orchestration)
            
            # Traverse the orchestration
            for key, value in orchestration.items():
                # Convert the value to string
                key_outputs = ""
                for k, output in value.items():
                    key_outputs += f"{k}: {output}; "
                
                # Create a new task
                new_task = BaseTreeTaskNode(
                    question=normalize_string(key), 
                    description=key_outputs, 
                    sub_task_depth=parent.sub_task_depth - 1,
                )
                # Link the new task to the parent task
                new_task.parent = parent
                # Add the new task to the parent task
                parent.sub_tasks[new_task.question] = new_task
                # If the sub task depth is 0, then set the task status to running
                if new_task.sub_task_depth == 0:
                    new_task.to_running()
                    
            # Format the task to ToDoTaskView
            view = ToDoTaskView(task=parent).format()
            # Return the user message
            return UserMessage(content=f"【成功】：任务创建成功。任务ToDo视图：\n{view}"), current_error
        
        except Exception as e:
            # Log the error
            logger.error(f"Error creating task: {e}")
            # Return the user message
            return UserMessage(content=f"【失败】：任务创建失败。错误信息：{e}"), current_error + 1
    
    async def reason(
        self, 
        target: TreeTaskNode, 
        max_idle_thinking: int = 1, 
        observe_args: dict[str, dict[str, Any]] = {}, 
    ) -> TreeTaskNode:
        """Reason about the task. This is the pre step of the planning in order to inference the real 
        and detailed requirements of the task. 
        
        Args:
            target (TreeTaskNode):
                The task to reason about.
            max_idle_thinking (int):
                The maximum number of idle thinking.
            observe_args (dict[str, dict[str, Any]]):
                The additional keyword arguments for observing the target. 
        
        Returns:
            TreeTaskNode: 
                The target after reasoning.
        """
        # Update system prompt to history
        message = SystemMessage(content=self.prompts["orchestrate_system"])
        target.update(message)
        
        # Observe the task
        observe = await self.agent.observe(target, **observe_args["orchestrate_think"])
        # Log the observation
        logger.info(f"Observe: \n{observe}")
        # Create a new message for the current observation
        message = UserMessage(content=self.prompts["orchestrate_think"])
        # Update the target with the user message
        target.update(message)
        # Create new user message with the observe
        message = UserMessage(content=message.content + f"\n\n## 观察\n以下是观察到的信息:\n{observe}")
        # Append the reason prompt to the task history
        target.update(message)
    
        while True:
            # Call for completion
            message: AssistantMessage = await self.agent.think(target.get_history())
            # Log the message
            if logger.level == "DEBUG":
                logger.debug(f"Full Assistant Message: \n{message}")
            else:
                logger.info(f"Assistant Message: \n{message.content}")
            # Record the completion message
            target.update(message)
            
            # Extract the orchestration blueprint from the task by regular expression
            blueprint = extract_by_label(message.content, "orchestration", "orchestrate", "blueprint")
            if blueprint != "":
                # Log the blueprint
                logger.info(f"Orchestration Blueprint: \n{blueprint}")
                # Update the blueprint to the global environment context
                self.agent.env.context = self.agent.env.context.create_next(blueprint=blueprint, task=target)
                # Stop the reason loop
                break
            else:
                # Update the current thinking
                current_thinking += 1
                # Check if the current thinking is greater than the max thinking
                if current_thinking > max_idle_thinking:
                    # Announce the idle thinking
                    message = UserMessage(content=f"【注意】：你已经达到了 {max_idle_thinking} 次思考上限，蓝图未找到，任务执行失败。")
                    # Append the message to the task history
                    target.update(message)
                    # Log the message
                    logger.critical(f"模型的连续 {max_idle_thinking} 次思考中没有找到规划蓝图，任务执行失败。")
                    # No more thinking is allowed, raise an error
                    raise RuntimeError("No orchestration blueprint was found in <orchestration> tags for 3 times thinking.")
                
                # No blueprint was found, create an error message
                message = UserMessage(
                    content=f"没有在<orchestration>标签中找到规划蓝图。请将你的规划放到<orchestration>标签中。你已经思考了 {current_thinking} 次，" \
                        f"在最多思考 {max_idle_thinking} 次后，任务会直接失败。下一步你必须给出规划蓝图，否则你将会被惩罚。",
                )
                # Append the error message to the task history
                target.update(message)
                # Log the message
                logger.warning(f"模型回复中没有找到规划蓝图，提醒模型重新思考。")
        
        return target
    
    async def reason_act(
        self, 
        target: Stateful, 
        react_think: str, 
        max_error_retry: int = 3, 
        current_error: int = 0, 
        max_idle_thinking: int = 1, 
        current_thinking: int = 0, 
        completion_config: dict[str, Any] = {}, 
        observe_args: dict[str, dict[str, Any]] = {}, 
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
            completion_config (dict[str, Any], optional, defaults to {}):
                The completion config of the workflow. 
            observe_args (dict[str, dict[str, Any]], optional, defaults to {}):
                The additional keyword arguments for observing the target. The following observe args must be provided:
                - "react_think": The observe args for the think stage.
                - "react_reflect": The observe args for the reflect stage.
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            tuple[Stateful, int, int]:
                The target, the current error counter and the current thinking counter.
        """
        # Observe the target
        observe = await self.agent.observe(target, **observe_args["react_think"])
        # Log the observe
        logger.info(f"Observe: \n{observe}")
        # Create new user message
        message = UserMessage(content=react_think)
        # Update the target with the user message
        target.update(message)
        # Create new user message with the observe
        message = UserMessage(content=message.content + f"\n\n## 观察\n以下是观察到的信息:\n{observe}")
        # Update the target with the user message
        target.update(message)
        
        # Think about the target
        message = await self.agent.think(target.get_history(), format_json=True)
        # Log the assistant message
        if logger.level == "DEBUG":
            logger.debug(f"Full Assistant Message: \n{message}")
        else:
            logger.info(f"Assistant Message: \n{message.content}")
        # Update the target with the assistant message
        target.update(message)

        # Create new tasks based on the orchestration json
        message, current_error = self.create_task(target, message.content, current_error)
        # Log the message
        logger.info(f"Create Task Message: \n{message.content}")
        # Update the target with the user message
        target.update(message)
        
        # Check if the current error is greater than the max error retry
        if current_error >= max_error_retry:
            # Set the target to error
            target.to_error()
            # Log the error
            logger.error(f"重试次数达到上限，错误重试次数: {current_error}/{max_error_retry}，`创建子任务`执行失败。")
            
        # Return the target, current error and current thinking
        return target, current_error, current_thinking
        
    async def run(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        prompts: dict[str, str] = {}, 
        completion_config: dict[str, Any] = {}, 
        observe_args: dict[str, dict[str, Any]] = {}, 
        running_checker: Callable[[Stateful], bool] = None, 
        *args, 
        **kwargs,
    ) -> TreeTaskNode:
        """Orchestrate the target. This workflow will not design any detailed plans, it will 
        only orchestrate the key objectives of the task. 

        Args:
            target (TreeTaskNode):
                The target to orchestrate.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of error retries.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of idle thinking.
            prompts (dict[str, str], optional, defaults to {}):
                The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
            completion_config (dict[str, Any], optional, defaults to {}):
                The completion config of the workflow. The following completion config are supported:
                - "tool_choice": The tool choice to use for the agent. 
                - "exclude_tools": The tools to exclude from the tool choice. 
            observe_args (dict[str, dict[str, Any]], optional, defaults to {}):
                The additional keyword arguments for observing the target. 
            running_checker (Callable[[Stateful], bool], optional, defaults to None):
                The checker to check if the workflow should be running.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.

        Returns:
            TreeTaskNode:
                The target after orchestrating.
        """
        # Check if the running checker is provided
        if running_checker is None:
            # Set the running checker to the default checker
            running_checker = lambda target: target.is_created()
        
        # Check if the target is running
        if running_checker(target):
            # Reason about the task and get the orchestration blueprint
            await self.reason(
                target=target, 
                max_idle_thinking=max_idle_thinking, 
                observe_args=observe_args, 
            )
        else:
            # Log the error
            logger.error("The target is not running, the workflow is not executed.")
            # Set the target to error
            target.to_error()
            # Return the target
            return target
        
        # Run the ReActFlow to create the tasks
        if running_checker(target):
            # Run the react flow
            target = await super().run(
                target=target, 
                max_error_retry=max_error_retry, 
                max_idle_thinking=max_idle_thinking, 
                prompts=prompts, 
                completion_config=completion_config, 
                observe_args=observe_args, 
                running_checker=running_checker, 
                *args, 
                **kwargs,
            )
        else:
            # Log the error
            logger.error("The target is not running after reasoning, the workflow is not executed.")
            # Set the target to error
            target.to_error()
            # Return the target
            return target
        
        # Return the target
        return target
    