from typing import Callable

from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import Agent, Context, TreeTaskNode, ReActFlow, CompletionConfig
from myagents.core.llms.config import BaseCompletionConfig
from myagents.core.messages import UserMessage, SystemMessage
from myagents.core.workflows.react import BaseReActFlow
from myagents.core.workflows.plan import PlanWorkflow
from myagents.core.utils.extractor import extract_by_label
from myagents.prompts.workflows.orchestrate import PROFILE


class BlueprintWorkflow(BaseReActFlow):
    """BlueprintWorkflow is a workflow for generating the blueprint of the task.
    """
    sub_workflows: dict[str, ReActFlow]
    
    async def run(
        self, 
        target: TreeTaskNode, 
        sub_task_depth: int = -1, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        completion_config: CompletionConfig = None, 
        running_checker: Callable[[TreeTaskNode], bool] = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """Run the workflow.
        
        Args:
            target (TreeTaskNode):
                The target to run the workflow on.
            sub_task_depth (int, optional, defaults to -1):
                The depth of the sub-task. If the sub-task depth is -1, then the sub-task depth will be 
                inferred from the target.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of times to idle thinking the agent.
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
            running_checker (Callable[[TreeTaskNode], bool], optional, defaults to None):
                The checker to check if the workflow should be running.
            **kwargs:
                The additional keyword arguments for running the workflow.
                
        Returns:
            TreeTaskNode:
                The target after running the workflow.
        """
        # Check if the running checker is provided
        if running_checker is None:
            # Set the running checker to the default running checker
            running_checker = lambda target: target.is_created()
        
        # Run the workflow
        return await super().run(
            target=target, 
            sub_task_depth=sub_task_depth, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config, 
            running_checker=running_checker, 
            **kwargs,
        )

    async def reason_act_reflect(
        self, 
        target: TreeTaskNode, 
        sub_task_depth: int = -1, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        completion_config: CompletionConfig = None, 
        running_checker: Callable[[TreeTaskNode], bool] = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """Reason and act on the target, and reflect on the target.
        
        Args:
            target (TreeTaskNode):
                The target to reason and act on.
            sub_task_depth (int, optional, defaults to -1):
                The depth of the sub-task. If the sub-task depth is -1, then the sub-task depth will be 
                the same as the depth of the target.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of times to idle thinking the agent. 
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
            running_checker (Callable[[TreeTaskNode], bool], optional, defaults to None):
                The checker to check if the workflow should be running.
            *args:
                The additional arguments for running the agent.
            **kwargs:
                The additional keyword arguments for running the agent. 
        """
        # Check if the running checker is provided
        if running_checker is None:
            # Set the running checker to the default running checker
            running_checker = lambda target: target.is_created()
        
        # Check if the target has history
        if len(target.get_history()) == 0:
            # Get the system prompt from the agent
            system_prompt = self.prompts["system_prompt"]
            # Update system prompt to history
            message = SystemMessage(content=system_prompt)
            target.update(message)
        
        # Error and idle thinking control
        current_thinking = 0
        current_error = 0
        
        # Run the workflow
        while running_checker(target):
        
            # === Reason Stage ===
            # Reason and act on the target
            target, error_flag, tool_call_flag = await self.reason_act(
                target=target, 
                sub_task_depth=sub_task_depth, 
                completion_config=completion_config, 
                **kwargs,
            )

            # === Extract Blueprint ===
            # Get the assistant message
            message = target.get_history()[-1]
            # Extract the orchestration blueprint from the task by regular expression
            blueprint = extract_by_label(message.content, "orchestration", "orchestrate", "blueprint")
            if blueprint != "":
                # Set tool_call_flag to True
                tool_call_flag = True
            
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
                    target.results += f"\n\n错误次数限制已达上限: {current_error}/{max_error_retry}，错误原因: {target.get_history()[-1].content}"
                    # Force the react loop to finish
                    break
            
            # === Reflect Stage ===
            # Reflect on the target
            target, finish_flag = await self.reflect(
                target=target, 
                completion_config=completion_config, 
                **kwargs,
            )
            # Check if the target is finished
            if finish_flag and blueprint != "":
                # Force the loop to break
                break
            
            # Check if the tool call flag is not set
            elif not tool_call_flag:
                # Increment the idle thinking counter
                current_thinking += 1
                # No blueprint was found, create an error message
                message = UserMessage(
                    content=f"没有在<orchestration>标签中找到规划蓝图。请将你的规划放到<orchestration>标签中。你已经思考了 {current_thinking} 次，" \
                        f"在最多思考 {max_idle_thinking} 次后，任务会直接失败。下一步你必须给出规划蓝图，否则你将会被惩罚。",
                )
                target.update(message)
                # Log the idle thinking message
                logger.warning(f"模型回复中没有找到规划蓝图，提醒模型重新思考。")
                # Check if the idle thinking counter is greater than the max idle thinking
                if current_thinking >= max_idle_thinking:
                    # Set the task status to error
                    target.to_error()
                    # Record the error as answer
                    target.results += f"模型的连续 {max_idle_thinking} 次思考中没有找到规划蓝图，进入错误状态。"
                    # Log the error message
                    logger.critical(f"模型的连续 {max_idle_thinking} 次思考中没有找到规划蓝图，任务执行失败。")
                    # Force the loop to break
                    break
            
        # === Update Context ===
        # Log the blueprint
        logger.info(f"Orchestration Blueprint: \n{blueprint}")
        # Update the blueprint to the global environment context
        self.context = self.context.create_next(blueprint=blueprint, task=target)
            
        return target
    
    async def reason_act(
        self, 
        target: TreeTaskNode, 
        sub_task_depth: int = -1, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> tuple[TreeTaskNode, bool, bool]:
        """Reason and act on the target.
        
        Args:
            target (TreeTaskNode):
                The target to reason and act on.
            sub_task_depth (int, optional, defaults to -1):
                The depth of the sub-task. If the sub-task depth is -1, then the sub-task depth will be 
                the same as the depth of the target.
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            tuple[TreeTaskNode, bool, bool]:
                The target, the error flag and the tool call flag.
        """
        # Check if the completion config is provided
        if completion_config is None:
            # Set the completion config to the default completion config
            completion_config = BaseCompletionConfig()
        else:
            # Ignore the tools
            completion_config.update(tools=[])
        
        # Initialize the error and tool call flag
        error_flag = False
        tool_call_flag = False
        
        # === Thinking ===
        # Observe the target
        observe = await self.agent.observe(
            target, 
            prompt=self.prompts["reason_act_prompt"], 
            observe_format=self.observe_formats["reason_act_format"]
        )
        # Log the observe
        logger.info(f"Observe: \n{observe[-1].content}")
        # Think about the target
        message = await self.agent.think(observe=observe, completion_config=completion_config)
        # Update the message to the target
        target.update(message)
        # Log the assistant message
        if logger.level == "DEBUG":
            logger.debug(f"{str(self.agent)}: \n{message}")
        else:
            logger.info(f"{str(self.agent)}: \n{message.content}")
        
        return target, error_flag, tool_call_flag


class OrchestrateFlow(PlanWorkflow):
    """This is use for Orchestrating the task. This workflow will not design any detailed plans, it will 
    only orchestrate the key objectives of the task. 
        
    Attributes:
        context (Context):
            The global context container of the workflow.
        tools (dict[str, FastMcpTool]):
            The tools can be used for the agent. 
        
        profile (str):
            The profile of the workflow.
        agent (Agent): 
            The agent that is used to orchestrate the task.
        prompts (dict[str, str]):
            The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
        observe_formats (dict[str, str]):
            The formats of the observation. The key is the observation name and the value is the format method name. 
        sub_workflows (dict[str, ReActFlow]):
            The sub-workflows of the workflow. The key is the name of the sub-workflow and the value is the 
            sub-workflow instance. 
    """
    # Context and tools
    context: Context
    tools: dict[str, FastMcpTool]
    # Basic information
    profile: str
    agent: Agent
    prompts: dict[str, str]
    observe_formats: dict[str, str]
    # Sub-worflows
    sub_workflows: dict[str, ReActFlow]
    # Need user check
    need_user_check: bool
    
    def __init__(
        self, 
        prompts: dict[str, str] = {}, 
        observe_formats: dict[str, str] = {}, 
        need_user_check: bool = False, 
        **kwargs,
    ) -> None:
        """Initialize the OrchestrateFlow.

        Args:
            profile (str):
                The profile of the workflow.
            prompts (dict[str, str]):
                The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
                The following keys are required:
                    "plan_system_prompt": The system prompt of the plan workflow.
                    "plan_reason_act_prompt": The reason and act prompt of the plan workflow.
                    "plan_reflect_prompt": The reflect prompt of the plan workflow.
                    "exec_system_prompt": The system prompt of the exec workflow.
                    "exec_reason_act_prompt": The reason and act prompt of the exec workflow.
                    "exec_reflect_prompt": The reflect prompt of the exec workflow.
            observe_formats (dict[str, str]):
                The formats of the observation. The key is the observation name and the value is the format method name. 
                The following keys are required:
                    "plan_reason_act": The format method name of the reason and act observation of the plan workflow.
                    "plan_reflect": The format method name of the reflect observation of the plan workflow.
                    "exec_reason_act": The format method name of the reason and act observation of the exec workflow.
                    "exec_reflect": The format method name of the reflect observation of the exec workflow.
            need_user_check (bool, optional, defaults to False):
                Whether to need the user to check the orchestration blueprint.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Create the sub-workflows
        sub_workflows = {
            "plan": BlueprintWorkflow(
                prompts={
                    "system_prompt": prompts["plan_system_prompt"], 
                    "reason_act_prompt": prompts["plan_reason_act_prompt"], 
                    "reflect_prompt": prompts["plan_reflect_prompt"], 
                }, 
                observe_formats={
                    "reason_act_format": observe_formats['plan_reason_act_format'], 
                    "reflect_format": observe_formats['plan_reflect_format'], 
                }, 
            ), 
        }
        
        # Prepare the prompts and observe formats for the exec workflow
        prompts = {
            "system_prompt": prompts["exec_system_prompt"], 
            "reason_act_prompt": prompts["exec_reason_act_prompt"], 
            "reflect_prompt": prompts["exec_reflect_prompt"], 
        }
        observe_formats = {
            "reason_act_format": observe_formats["exec_reason_act_format"], 
            "reflect_format": observe_formats["exec_reflect_format"], 
        }
        self.need_user_check = need_user_check
        
        # Initialize the parent class
        super().__init__(
            profile=PROFILE, 
            prompts=prompts, 
            observe_formats=observe_formats, 
            sub_workflows=sub_workflows, 
            **kwargs,
        )
    
    async def reason(
        self, 
        target: TreeTaskNode, 
        sub_task_depth: int = -1, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        completion_config: CompletionConfig = None, 
        running_checker: Callable[[TreeTaskNode], bool] = None, 
        **kwargs, 
    ) -> tuple[TreeTaskNode, bool]:
        """Reason about the task. This is the pre step of the planning in order to inference the real 
        and detailed requirements of the task. 
        
        Args:
            target (TreeTaskNode):
                The task to reason about.
            sub_task_depth (int, optional, defaults to -1):
                The depth of the sub-task. If the sub-task depth is -1, then the sub-task depth will be 
                the same as the depth of the target.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of error retries.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of idle thinking.
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow.
            running_checker (Callable[[TreeTaskNode], bool], optional, defaults to None):
                The checker to check if the workflow should be running.
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            TreeTaskNode: 
                The target after reasoning.
            bool:
                The flag to check if the orchestration blueprint is valid.
        """
        # Call the blueprint workflow
        target = await self.sub_workflows["plan"].reason_act_reflect(
            target=target, 
            sub_task_depth=sub_task_depth, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config, 
            running_checker=running_checker, 
            **kwargs,
        )
        
        try:
            # Get the blueprint from the context
            blueprint = self.sub_workflows["plan"].context.get("blueprint")
            # Check if the blueprint is valid
            if blueprint == "":
                # Log the error
                logger.error("The blueprint is not valid.")
                # Return the target and the flag
                return target, False
            
            # Update the blueprint to the environment context
            self.agent.env.context = self.agent.env.context.create_next(blueprint=blueprint, task=target)
            # Return the target and the flag
            return target, True
        except Exception as e:
            # Log the error
            logger.error(f"Error updating the blueprint to the environment context: {e}")
            # Return the target and the flag
            return target, False
    
    async def reason_act(
        self, 
        target: TreeTaskNode, 
        sub_task_depth: int = -1, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> tuple[TreeTaskNode, bool, bool]:
        """Reason and act on the target.
        
        Args:
            target (TreeTaskNode):
                The target to reason and act on.
            sub_task_depth (int, optional, defaults to -1):
                The depth of the sub-task. If the sub-task depth is -1, then the sub-task depth will be 
                the same as the depth of the target.
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            tuple[TreeTaskNode, bool, bool]:
                The target, the error flag and the tool call flag.
        """
        # Check if the completion config is provided
        if completion_config is None:
            # Set the completion config to the default completion config
            completion_config = BaseCompletionConfig()
        
        # Initialize the error and tool call flag
        error_flag = False
        tool_call_flag = False
        
        # Observe the target
        observe = await self.agent.observe(
            target, 
            prompt=self.prompts["reason_act_prompt"], 
            observe_format=self.observe_formats["reason_act_format"], 
        )
        # Log the observe
        logger.info(f"Observe: \n{observe[-1].content}")
        
        # Think about the target
        message = await self.agent.think(
            observe=observe, 
            completion_config=completion_config, 
        )
        # Update the message to the target
        target.update(message)
        # Log the assistant message
        if logger.level == "DEBUG":
            logger.debug(f"{str(self.agent)}: \n{message}")
        else:
            logger.info(f"{str(self.agent)}: \n{message.content}")

        # Create new tasks based on the orchestration json
        message, error_flag = self.create_task(
            parent=target, 
            orchestrate_json=message.content, 
            sub_task_depth=sub_task_depth, 
        )    # BUG: 这里message.content可能为None
        # Log the message
        logger.info(f"Create Task Message: \n{message.content}")
        # Update the target with the user message
        target.update(message)
        
        # Return the target, error flag and tool call flag
        return target, error_flag, tool_call_flag
        
    async def run(
        self, 
        target: TreeTaskNode, 
        sub_task_depth: int = -1, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        completion_config: CompletionConfig = None, 
        running_checker: Callable[[TreeTaskNode], bool] = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """Orchestrate the target. This workflow will not design any detailed plans, it will 
        only orchestrate the key objectives of the task. 

        Args:
            target (TreeTaskNode):
                The target to orchestrate.
            sub_task_depth (int, optional, defaults to -1):
                The depth of the sub-task. If the sub-task depth is -1, then the sub-task depth will be 
                the same as the depth of the target.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of error retries.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of idle thinking.
            completion_config (CompletionConfig, optional, defaults to None):
                The completion config of the workflow. 
            running_checker (Callable[[TreeTaskNode], bool], optional, defaults to None):
                The checker to check if the workflow should be running. 
            **kwargs:
                The keyword arguments to be passed to the parent class.

        Returns:
            TreeTaskNode:
                The target after orchestrating.
        """
        # Check if the running checker is provided
        if running_checker is None:
            # Set the running checker to the default running checker
            running_checker = lambda target: target.is_created()
        
        # Check if the target is running
        while running_checker(target):
            
            # Reason about the task and get the orchestration blueprint
            target, blueprint_valid = await self.reason(
                target=target, 
                sub_task_depth=sub_task_depth, 
                max_error_retry=max_error_retry, 
                max_idle_thinking=max_idle_thinking, 
                completion_config=completion_config, 
                running_checker=running_checker, 
                **kwargs, 
            )
            
            # Check if the orchestration blueprint is valid
            if blueprint_valid:
                # Log the valid blueprint
                logger.info("Orchestration blueprint is valid.")
                
                # Check if the need user check is set
                if self.need_user_check:
                    raise NotImplementedError("User check is not implemented yet.")
                    # Call the proxy agent of the environment for response
                    message = await self.agent.env.call_agent()
                    # Update the message to the target
                    target.update(message)
                    # Log the message
                    logger.info(f"User Check Message: \n{message.content}")
                    # Check if the user has checked the orchestration blueprint
                    if message.content == "yes":
                        # Break the loop
                        break
                
                else:
                    # Break the loop
                    break
        
        # Run the ReActFlow to create the tasks
        if running_checker(target):
            # Run the react flow
            target = await super().run(
                target=target, 
                sub_task_depth=sub_task_depth, 
                max_error_retry=max_error_retry, 
                max_idle_thinking=max_idle_thinking, 
                completion_config=completion_config, 
                running_checker=running_checker, 
                **kwargs,
            )
        else:
            # Log the error
            logger.error("The target is not running after reasoning, the workflow is not executed.")
            # Set the target to error
            target.to_error()   # NOTE: Blueprint 提取失败且 reflect 没有检查到，则会导致 target 被设为 error
            # Return the target
            return target
        
        # Return the target
        return target
