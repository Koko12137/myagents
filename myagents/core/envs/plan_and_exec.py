from asyncio import Semaphore
from typing import Union

from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.schemas.messages import AssistantMessage, UserMessage, SystemMessage, ToolCallResult
from myagents.core.interface import Agent, TreeTaskNode, Workspace, CompletionConfig
from myagents.core.agents import AgentType
from myagents.core.tasks import DocumentTaskView, ToDoTaskView
from myagents.core.envs.base import BaseEnvironment, EnvironmentStatus


REQUIRED_AGENTS = [
    AgentType.ORCHESTRATE, 
    AgentType.TREE_REACT, 
]


class PlanAndExecEnv(BaseEnvironment):
    """
    PlanAndExecEnv is a environment for splitting a task into sub-tasks and executing the sub-tasks.
    """
    # Basic information
    uid: str
    name: str
    profile: str
    prompts: dict[str, str]
    required_agents: list[AgentType]
    # Core components
    agents: dict[str, Agent]
    agent_type_map: dict[AgentType, list[str]]
    agent_type_semaphore: dict[AgentType, Semaphore]
    # Tools Mixin
    tools: dict[str, FastMcpTool]
    # Workspace
    workspace: Workspace
    # Stateful Mixin
    status: EnvironmentStatus
    history: list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]
    
    def __init__(
        self, 
        name: str, 
        profile: str, 
        prompts: dict[str, str], 
        required_agents: list[AgentType], 
        **kwargs, 
    ) -> None:
        """Initialize the BaseEnvironment.
        
        Args:
            name (str):
                The name of the environment.
            profile (str):
                The profile of the environment.
            prompts (dict[str, str]):
                The prompts of the environment. The key is the prompt name and the value is the prompt content. 
            required_agents (list[AgentType]):
                The required agents to work on the environment. The agents in the list must be registered to the environment. 
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Check if the required agents are in the required agents list
        for agent_type in required_agents:
            if agent_type not in REQUIRED_AGENTS:
                # Append the agent type to the required agents list
                required_agents.append(agent_type)
        
        # Initialize the parent class
        super().__init__(
            name=name, 
            profile=profile, 
            prompts=prompts, 
            required_agents=required_agents, 
            **kwargs,
        )
            
        # Check if the required prompts are in the prompts list
        for prompt_name in ["orchestrate_prompt", "execute_prompt", "error_prompt"]:
            if prompt_name not in prompts:
                # Log the error
                logger.error(f"Prompt `{prompt_name}` is not in the prompts list.")
                # Raise the error
                raise RuntimeError(f"Prompt `{prompt_name}` is not in the prompts list.")
        
    async def schedule(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """Override the schedule method of the react workflow.

        Args:
            target (TreeTaskNode):
                The target to plan and execute. 
            max_error_retry (int):
                The maximum number of error retries.
            max_idle_thinking (int):
                The maximum number of idle thinking.
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            **kwargs:
                The keyword arguments to be passed to the parent class.

        Returns:
            TreeTaskNode:
                The target after scheduling. 
        
        Raises:
            ValueError:
                If the target is not created.
        """
        # Get the error prompt from the agent
        error_prompt = self.prompts["error_prompt"]
        # Record the current error retry count
        current_error_retry = 0
        
        while not target.is_finished():
            
            # Check if the target is created, if True, then plan the task
            if target.is_created() and (
                # Check if the target has sub-tasks and any of the sub-tasks are cancelled
                (len(target.sub_tasks) != 0 and any(sub_task.is_cancelled() for sub_task in target.sub_tasks.values())) or
                # Check if the target has no sub-tasks
                (len(target.sub_tasks) == 0)
            ):
                # Check if any of the sub-tasks are cancelled
                if any(sub_task.is_cancelled() for sub_task in target.sub_tasks.values()):
                    # Get the current result
                    current_result = DocumentTaskView(target).format()
                    # Create a new user message to record the error and the current result
                    message = UserMessage(content=error_prompt.format(
                        error_retry=current_error_retry, 
                        max_error_retry=max_error_retry, 
                        error_reason=target.results,
                        current_result=current_result,
                    ))
                    target.update(message)
                
                    # Get all the unfinished sub-tasks
                    unfinished_sub_tasks = [sub_task for sub_task in target.sub_tasks.values() if sub_task.is_cancelled()]
                    # Delete the unfinished sub-tasks
                    for sub_task in unfinished_sub_tasks:
                        del target.sub_tasks[sub_task.name]
                        # Log the deleted sub-task
                        logger.error(f"删除已取消的子任务: \n{ToDoTaskView(sub_task).format()}")
                
                # Get the orchestrate prompt from the agent
                orchestrate_prompt = self.prompts["orchestrate_prompt"]
                # Create a new user message to record the orchestrate prompt
                message = UserMessage(content=orchestrate_prompt)
                # Update the environment history
                self.update(message)
                
                # Plan the task
                message = await self.call_agent(
                    agent_type=AgentType.ORCHESTRATE, 
                    target=target, 
                    max_error_retry=max_error_retry, 
                    max_idle_thinking=max_idle_thinking, 
                    completion_config=completion_config, 
                    **kwargs,
                )
                # Update the environment history
                self.update(message)
                # Log the answer
                logger.info(f"Agent Response: \n{message.content}")
            
            elif (target.is_created() and len(target.sub_tasks) > 0) or target.is_running():
                # Execute the task
                target = await self.execute(
                    target=target, 
                    max_error_retry=max_error_retry, 
                    max_idle_thinking=max_idle_thinking, 
                    completion_config=completion_config, 
                    **kwargs,
                )
                
            elif target.is_finished():
                # Break the loop
                break
            
            elif target.is_error():
                # Rollback the target to created
                target.to_created()
                # Increment the error retry count
                current_error_retry += 1

                # Check if the error retry count is greater than the max error retry
                if current_error_retry > max_error_retry:
                    # Log the error
                    logger.error(f"错误次数 {current_error_retry} 超过最大错误次数 {max_error_retry}，任务终止。")
                    # Set the target status to cancelled
                    target.to_cancelled()
                    # Break the loop
                    break
                
                # Set the sub task status to cancelled if not finished
                for sub_task in target.sub_tasks.values():
                    if not sub_task.is_finished():
                        sub_task.to_cancelled()
                
                # Log the cancelled sub-task
                logger.error(f"取消所有未执行或执行失败的子任务: \n{ToDoTaskView(target).format()}")
                
            elif target.is_cancelled():
                # Log the cancelled target
                logger.error(f"目标已被取消: \n{ToDoTaskView(target).format()}")
                # Break the loop
                break
            
            else:
                # Log the error
                logger.critical(f"Invalid target status in plan and exec workflow: {target.get_status()}")
                # Raise the error
                raise RuntimeError(f"Invalid target status in plan and exec workflow: {target.get_status()}")
            
        # Return the target
        return target
    
    async def execute(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int, 
        max_idle_thinking: int, 
        completion_config: CompletionConfig = None, 
        **kwargs,
    ) -> TreeTaskNode:
        """Deep first execute the task. This is the post step of the planning in order to execute the task. 
        
        Args:
            target (TreeTaskNode):
                The task to deep first execute.
            max_error_retry (int):
                The maximum number of error retries.
            max_idle_thinking (int):
                The maximum number of idle thinking. 
            completion_config (CompletionConfig):
                The completion config of the workflow. 
            **kwargs:
                The additional keyword arguments for running the agent.
                
        Returns:
            TreeTaskNode:
                The target after deep first executing.
        """
        if len(target.sub_tasks) > 0:
            sub_task = target.sub_tasks[list(target.sub_tasks)[0]]
        else:
            sub_task = target
        
        # Traverse all the sub-tasks
        while sub_task != target:
            
            # Check if the sub-task is created, if True, there must be some error in the sub-task
            if sub_task.is_created():
                sub_task = await self.schedule(
                    target=sub_task, 
                    max_error_retry=max_error_retry, 
                    max_idle_thinking=max_idle_thinking, 
                    completion_config=completion_config, 
                    **kwargs,
                )
                
            # Check if the sub-task is running, if True, then act the sub-task
            elif sub_task.is_running():
                # Log the sub-task
                logger.info(f"执行子任务: \n{ToDoTaskView(sub_task).format()}")
                # Act the sub-task
                sub_task = await self.execute(
                    sub_task, 
                    max_error_retry=max_error_retry, 
                    max_idle_thinking=max_idle_thinking, 
                    completion_config=completion_config, 
                    **kwargs,
                )

            # Check if the sub-task is failed, if True, then retry the sub-task
            elif sub_task.is_error():
                # Log the error
                logger.error(f"子任务执行失败: \n{ToDoTaskView(sub_task).format()}")
                # Rollback the target to error
                target.to_error()
                # Return the target
                return target
            
            # Check if the sub-task is cancelled, if True, set the parent task status to created and stop the traverse
            elif sub_task.is_cancelled():
                # Log the error
                logger.error(f"子任务已被取消: \n{ToDoTaskView(sub_task).format()}")
                # Rollback the target to error
                target.to_error()
                # Return the target
                return target
            
            # Check if the sub-task is finished, if True, then summarize the result of the sub-task
            elif sub_task.is_finished():
                # Log the finished sub-task
                logger.info(f"子任务执行完成: \n{ToDoTaskView(sub_task).format()}")
                # Get the next unfinished sub-task
                sub_task = sub_task.next
            
            # ELSE, the sub-task is not created, running, failed, cancelled, or finished, it is a critical error
            else:
                # The sub-task is not created, running, failed, cancelled, or finished, then raise an error
                raise RuntimeError(f"The status of the sub-task is invalid in action flow: {sub_task.get_status()}")
            
        # Post traverse the task
        # All the sub-tasks are finished, then reason, act and reflect on the task
        if all(sub_task.is_finished() for sub_task in target.sub_tasks.values()):
            # Get the execute prompt from the agent
            execute_prompt = self.prompts["execute_prompt"]
            # Create a new user message to record the execute prompt
            message = UserMessage(content=execute_prompt)
            # Update the environment history
            self.update(message)
            
            # Call the parent class to reason, act and reflect
            message = await self.call_agent(
                agent_type=AgentType.TREE_REACT, 
                target=target, 
                max_error_retry=max_error_retry, 
                max_idle_thinking=max_idle_thinking, 
                completion_config=completion_config, 
                **kwargs,
            )
            # Update the environment history
            self.update(message)
            # Log the answer
            logger.info(f"Agent Response: \n{message.content}")
            
        return target 
