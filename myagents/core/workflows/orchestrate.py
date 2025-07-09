import json
from typing import Callable

from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from json_repair import repair_json

from myagents.core.interface import Agent, Stateful, Context
from myagents.core.messages import AssistantMessage, UserMessage, SystemMessage, ToolCallResult
from myagents.core.workflows.react import ReActFlow
from myagents.core.tasks import BaseTreeTaskNode
from myagents.core.utils.extractor import extract_by_label
from myagents.core.utils.strings import normalize_string
from myagents.prompts.workflows.orchestrate import PROFILE, SYSTEM_PROMPT, THINK_PROMPT, ACTION_PROMPT


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
        action_prompt: str = "", 
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
            action_prompt (str, optional, defaults to ""):
                The action prompt of the workflow.
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
            "react_think": action_prompt if action_prompt != "" else ACTION_PROMPT,
        })
        
    async def post_init(self) -> None:
        """Post init is the method that will be called after the initialization of the workflow.
        
        This method will be called after the initialization of the workflow.
        """
        # Call the parent post init method
        await super().post_init()
        
        # Register the create task tool
        @self.register_tool("create_task")
        async def create_task(orchestration: str) -> ToolCallResult:
            """
            创建一个新的任务，并将其添加到当前任务的子任务中。
            
            Args:
                orchestration (str): 
                    当前任务的规划蓝图。应该是一个json格式的输入，下面是输入举例:
                    ```json
                    {
                        "任务目标1": {
                            "关键产出 1.1": "关键产出1.1的描述",
                            "关键产出 1.2": "关键产出1.2的描述",
                            ...
                        },
                        "任务目标2": {
                            "关键产出 2.1": "关键产出2.1的描述",
                            "关键产出 2.2": "关键产出2.2的描述",
                            ...
                        }
                    }
                    ```
                
            Returns:
                ToolCallResult: 
                    创建子任务的工具调用结果。
            """
            # Get the parent task from the context
            parent = self.context.get("task")
            # Get the function call details
            tool_call = self.context.get("tool_call")
            
            # Repair the json
            orchestration = repair_json(orchestration)
            # Parse the json
            orchestration = json.loads(orchestration)
            
            # Traverse the orchestration
            for key, value in orchestration.items():
                # Create a new task
                new_task = BaseTreeTaskNode(
                    question=normalize_string(key), 
                    description=str(value), 
                )
                # Link the new task to the parent task
                new_task.parent = parent
                # Add the new task to the parent task
                parent.sub_tasks[new_task.question] = new_task
            
            # Create a new tool call result
            tool_call_result = ToolCallResult(
                tool_call_id=tool_call.id,
                content="任务创建成功", 
            )
            return tool_call_result
        
    async def __reason(
        self, 
        target: Stateful, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [], 
        *args, 
        **kwargs,
    ) -> Stateful:
        """Reason about the task. This is the pre step of the planning in order to inference the real 
        and detailed requirements of the task. 
        
        Args:
            target (Stateful):
                The task to reason about.
            max_error_retry (int):
                The maximum number of error retries.
            max_idle_thinking (int):
                The maximum number of idle thinking.
            tool_choice (str):
                The tool choice of the agent.
            exclude_tools (list[str]):
                The tools to exclude from the agent.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.

        Returns:
            Stateful: 
                The target after reasoning.
        """
        # Update system prompt to history
        message = SystemMessage(content=self.prompts["orchestrate_system"])
        target.update(message)
        
        # Observe the task
        observe = await self.agent.observe(target)
        # Log the observation
        logger.info(f"Observe: \n{observe}")
        # Create a new message for the current observation
        message = UserMessage(content=self.prompts["orchestrate_think"].format(observe=observe))
        # Append the reason prompt to the task history
        target.update(message)
    
        while True:
            # Call for completion
            message: AssistantMessage = await self.agent.think(target.get_history())
            # Log the message
            logger.info(f"Assistant Message: \n{message.content}")
            # Record the completion message
            target.update(message)
            
            # Extract the orchestration blueprint from the task by regular expression
            blueprint = extract_by_label(message.content, "orchestration", "orchestrate", "blueprint")
            if blueprint != "":
                # Log the blueprint
                logger.info(f"Orchestration Blueprint: \n{blueprint}")
                # Update the blueprint to the global workflow context
                self.context = self.context.create_next(blueprint=blueprint, task=target)
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
        """Orchestrate the target. This workflow will not design any detailed plans, it will 
        only orchestrate the key objectives of the task. 

        Args:
            target (Stateful):
                The target to orchestrate.
            max_error_retry (int):
                The maximum number of error retries.
            max_idle_thinking (int):
                The maximum number of idle thinking.
            tool_choice (str):
                The tool choice of the agent.
            exclude_tools (list[str]):
                The tools to exclude from the agent.
            running_checker (Callable[[Stateful], bool]):
                The checker to check if the workflow should be running.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.

        Returns:
            Stateful:
                The target after orchestrating.
        """
        # Check if the running checker is provided
        if running_checker is None:
            # Set the running checker to the default checker
            running_checker = lambda target: target.is_created()
        
        # Check if the target is running
        if running_checker(target):
            # Reason about the task and get the orchestration blueprint
            await self.__reason(
                target, 
                max_error_retry, 
                max_idle_thinking, 
                tool_choice, 
                exclude_tools, 
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
        
        # Run the ReActFlow to create the tasks
        if running_checker(target):
            # Designate a tool choice for the react flow
            tool_choice = "create_task"
            
            await super().run(
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
            logger.error("The target is not running after reasoning, the workflow is not executed.")
            # Set the target to error
            target.to_error()
            # Return the target
            return target
        
        # Return the target
        return target
    