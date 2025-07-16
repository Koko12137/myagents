import json
from enum import Enum
from typing import Callable, Any

from json_repair import repair_json
from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import Agent, Stateful, Context, TreeTaskNode
from myagents.core.messages import AssistantMessage, UserMessage, SystemMessage
from myagents.core.workflows.react import ReActFlow
from myagents.core.tasks import BaseTreeTaskNode, ToDoTaskView
from myagents.core.utils.extractor import extract_by_label
from myagents.core.utils.strings import normalize_string
from myagents.prompts.workflows.orchestrate import PROFILE, SYSTEM_PROMPT


class OrchestrateStage(Enum):
    """The stage of the orchestrate workflow.
    - REASON: The reason stage.
    - REASON_ACT: The reason and act stage.
    - REFLECT: The reflect stage.
    """
    # Reason init stage
    REASON_INIT = 0
    # Reason stage
    REASON = 1
    # ReAct init stage
    REACT_INIT = 2
    # Reason and act stage
    REASON_ACT = 3
    # Reflect stage
    REFLECT = 4


class OrchestrateFlow(ReActFlow):
    """This is use for Orchestrating the task. This workflow will not design any detailed plans, it will 
    only orchestrate the key objectives of the task. 
        
    Attributes:
        profile (str):
            The profile of the workflow.
        agent (Agent): 
            The agent that is used to orchestrate the task.
        context (Context):
            The global context container of the workflow.
        tools (dict[str, FastMcpTool]):
            The tools can be used for the agent. 
        stage (Enum):
            The stage of the workflow.
    """
    # Basic information
    profile: str
    agent: Agent
    # Context and tools
    context: Context
    tools: dict[str, FastMcpTool]
    # Workflow stage
    stage: Enum

    def __init__(
        self, 
        profile: str = PROFILE, 
        *args, 
        **kwargs,
    ) -> None:
        """Initialize the OrchestrateFlow.

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
    
    def create_task(
        self, 
        parent: TreeTaskNode, 
        orchestration: str, 
    ) -> tuple[UserMessage, bool]:
        """Create a new task based on the orchestration blueprint.
        
        Args:
            parent (TreeTaskNode):
                The parent task to create the new task.
            orchestration (str):
                The orchestration blueprint to create the new task.
                
        Returns:
            UserMessage:
                The user message after creating the new task.
            bool:
                The error flag.
        """
        try:
            # Repair the json
            orchestration = repair_json(orchestration)
            # Parse the orchestration
            orchestration: dict[str, dict[str, str]] = json.loads(orchestration)
            
            # Traverse the orchestration
            for uid, value in orchestration.items():
                # Convert the value to string
                key_outputs = ""
                for output in value['关键产出']:
                    key_outputs += f"{output}; "
                
                # Create a new task
                new_task = BaseTreeTaskNode(
                    uid=uid, 
                    objective=normalize_string(value['目标描述']), 
                    key_results=key_outputs, 
                    sub_task_depth=parent.sub_task_depth - 1,
                )
                # Link the new task to the parent task
                new_task.parent = parent
                # Add the new task to the parent task
                parent.sub_tasks[uid] = new_task
                # If the sub task depth is 0, then set the task status to running
                if new_task.sub_task_depth == 0:
                    new_task.to_running()
                    
            # Format the task to ToDoTaskView
            view = ToDoTaskView(task=parent).format()
            # Return the user message
            return UserMessage(content=f"【成功】：任务创建成功。任务ToDo视图：\n{view}"), False
        
        except Exception as e:
            # Log the error
            logger.error(f"Error creating task: {e}")
            # Return the user message
            return UserMessage(content=f"【失败】：任务创建失败。错误信息：{e}"), True
    
    async def reason(
        self, 
        target: TreeTaskNode, 
        max_idle_thinking: int = 1, 
        init_stage: Enum = OrchestrateStage.REASON_INIT, 
        to_stage: Enum = OrchestrateStage.REASON, 
    ) -> TreeTaskNode:
        """Reason about the task. This is the pre step of the planning in order to inference the real 
        and detailed requirements of the task. 
        
        Args:
            target (TreeTaskNode):
                The task to reason about.
            max_idle_thinking (int, optional):
                The maximum number of idle thinking.
            init_stage (Enum, optional):
                The stage to initialize the workflow.
            to_stage (Enum, optional):
                The stage to reason about.
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            TreeTaskNode: 
                The target after reasoning.
        """
        # Update the stage
        self.stage = to_stage
        
        # Get the system prompt from the agent
        system_prompt = self.agent.prompts[init_stage]
        # Update system prompt to history
        message = SystemMessage(content=system_prompt)
        target.update(message)
        
        # Observe the task
        observe = await self.agent.observe(target)
        # Log the observation
        logger.info(f"Observe: \n{observe}")
        # Create a new message for the current observation
        message = UserMessage(content=observe)
        # Update the target with the user message
        target.update(message)
        
        # Reason loop
        while True:
            # Call for completion
            message: AssistantMessage = await self.agent.think(target.get_history())
            # Log the message
            if logger.level == "DEBUG":
                logger.debug(f"{str(self.agent)}: \n{message}")
            else:
                logger.info(f"{str(self.agent)}: \n{message.content}")
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
        to_stage: Enum = OrchestrateStage.REASON_ACT, 
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
                The completion config of the workflow. 
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
        
        # Observe the target
        observe = await self.agent.observe(target)
        # Log the observe
        logger.info(f"Observe: \n{observe}")
        # Create new user message
        message = UserMessage(content=observe)
        # Update the target with the user message
        target.update(message)
        
        # Think about the target
        message = await self.agent.think(target.get_history(), format_json=True)
        # Log the assistant message
        if logger.level == "DEBUG":
            logger.debug(f"{str(self.agent)}: \n{message}")
        else:
            logger.info(f"{str(self.agent)}: \n{message.content}")
        # Update the target with the assistant message
        target.update(message)

        # Create new tasks based on the orchestration json
        message, error_flag = self.create_task(target, message.content)
        # Log the message
        logger.info(f"Create Task Message: \n{message.content}")
        # Update the target with the user message
        target.update(message)
        
        # Return the target, error flag and tool call flag
        return target, error_flag, tool_call_flag
        
    async def run(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        completion_config: dict[str, Any] = {}, 
        running_checker: Callable[[Stateful], bool] = None, 
        *args, 
        **kwargs,
    ) -> TreeTaskNode:
        """Orchestrate the target. This workflow will not design any detailed plans, it will 
        only orchestrate the key objectives of the task. 

        Args:
            target (TreeTaskNode):
                The target to orchestrate.
            max_error_retry (int, optional):
                The maximum number of error retries.
            max_idle_thinking (int, optional):
                The maximum number of idle thinking.
            completion_config (dict[str, Any], optional):
                The completion config of the workflow. The following completion config are supported:
                - "tool_choice": The tool choice to use for the agent. 
                - "exclude_tools": The tools to exclude from the tool choice. 
            running_checker (Callable[[Stateful], bool], optional):
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
            )
        else:
            # Log the error
            logger.error("The target is not running, the workflow is not executed.")
            # Set the target to error
            target.to_error()
            # Return the target
            return target
        
        # Prepare valid stages
        valid_stages = {
            "init": OrchestrateStage.REACT_INIT, 
            "reason_act": OrchestrateStage.REASON_ACT, 
            "reflect": OrchestrateStage.REFLECT, 
        }
        
        # Run the ReActFlow to create the tasks
        if running_checker(target):
            # Run the react flow
            target = await super().run(
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
            logger.error("The target is not running after reasoning, the workflow is not executed.")
            # Set the target to error
            target.to_error()   # BUG: 这里不知道为什么有时候会被设为error，检查前面出错时重试后是否会走到这里
            # Return the target
            return target
        
        # Return the target
        return target
    