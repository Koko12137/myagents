import json
from asyncio import Semaphore
from collections import OrderedDict
from typing import Union

from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.messages import AssistantMessage, UserMessage, SystemMessage, ToolCallResult
from myagents.core.interface import Agent, TreeTaskNode, Context
from myagents.core.agents import AgentType
from myagents.core.tasks import BaseTreeTaskNode, ToDoTaskView, JsonTaskView
from myagents.core.llms.config import BaseCompletionConfig
from myagents.core.envs.base import BaseEnvironment, EnvironmentStatus
from myagents.prompts.envs.query import (
    NAME, 
    PROFILE, 
    QUERY_ORCHESTRATE_PROMPT, 
)


REQUIRED_AGENTS = [AgentType.ORCHESTRATE]


class Orchestrate(BaseEnvironment):
    """Orchestrate is the environment for the multi-turn orchestrate and answer the question.
    
    Attributes:
        uid (str):
            The unique identifier of the environment. 
        name (str):
            The name of the environment.
        profile (str):
            The profile of the environment. 
        leader (Agent):
            The leader agent of the environment. 
        agents (dict[str, Agent]):
            The agents in the environment. The key is the agent name and the value is the agent. 
        required_agents (list[AgentType]):
            The agents in the list must be registered to the environment. 
        agent_type_map (dict[AgentType, list[str]]):
            The map of the agent type to the agent name. The key is the agent type and the value is the agent name list. 
        agent_type_semaphore (dict[AgentType, Semaphore]):
            The semaphore of the agent type. The key is the agent type and the value is the semaphore. 
        tools (dict[str, FastMcpTool]):
            The tools that can be used to modify the environment. The key is the tool name and the value is the tool. 
        context (Context):
            The context of the environment.
        status (EnvironmentStatus):
            The status of the environment.
        history (list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]):
            The history messages of the environment. 
        tasks (OrderedDict[str, Task]):
            The tasks of the environment. The key is the task question and the value is the task. 
    """
    # Basic information
    uid: str
    name: str
    profile: str
    prompts: dict[str, str]
    required_agents: list[AgentType]
    # Core components
    leader: Agent
    agents: dict[str, Agent]
    agent_type_map: dict[AgentType, list[str]]
    agent_type_semaphore: dict[AgentType, Semaphore]
    # Tools
    tools: dict[str, FastMcpTool]
    # Context
    context: Context
    # Status and history
    status: EnvironmentStatus
    history: list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]
    # Additional components
    tasks: OrderedDict[str, TreeTaskNode]
    answers: OrderedDict[str, str]
    
    def __init__(
        self, 
        profile: str = PROFILE, 
        orchestration_prompt: str = QUERY_ORCHESTRATE_PROMPT, 
        **kwargs,
    ) -> None:
        """Initialize the Orchestrate environment.
        
        Args:
            profile (str, optional, defaults to PROFILE):
                The profile of the environment.
            orchestration_prompt (str, optional, defaults to QUERY_ORCHESTRATION_PROMPT):
                The orchestration prompt of the environment.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        super().__init__(
            name=NAME, 
            profile=profile,
            prompts={
                "orchestration": orchestration_prompt, 
            }, 
            required_agents=REQUIRED_AGENTS, 
            **kwargs,
        )
        
        # Initialize the tasks
        self.tasks = OrderedDict()
        # Initialize the answers
        self.answers = OrderedDict()
        # Post initialize
        self.post_init()
        
    def __str__(self) -> str:
        """Get the string representation of the environment.
        """
        return f"Orchestrate(name={self.name}, profile={self.profile}, status={self.status})"
    
    def __repr__(self) -> str:
        """Get the string representation of the environment.
        """
        return self.__str__()
        
    def post_init(self) -> None:
        """Post initialize the Orchestrate environment.
        """
        pass
    
    async def observe(self, format: str = "none", **kwargs) -> str:
        """Observe the environment.
        
        Returns:
            str:
                The observation of the environment.
        """
        pass
                
    async def run(
        self, 
        question: str, 
        description: str, 
        sub_task_depth: int = 3, 
        completion_config: BaseCompletionConfig = None, 
        **kwargs,
    ) -> str:
        """Run the orchestrate for the question.
        
        Args:
            question (str):
                The question to be answered.
            description (str):
                The detail information and limitation of the task.
            sub_task_depth (int):
                The max number of layers of sub-question layers that can be split from the question. 
                The sub task depth should be greater than 0 and less than 5.
            completion_config (BaseCompletionConfig, optional):
                The completion config of the environment.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        Returns:
            tuple[str, str, str]:
                The blueprint, todo view and json view of the task. 
                
        Raises:
            ValueError:
                The detail level is not valid.
        """
        # Check the required agents are registered
        for agent_type in self.required_agents:
            # Try to get the agent type from the agent type map
            agent_names = self.agent_type_map.get(agent_type, [])
            # Check if the agent type is registered
            if len(agent_names) == 0 or agent_type not in self.agent_type_map:
                raise ValueError(f"Agent type `{agent_type}` is not registered. Please register the agent type to the environment.")
        
        # Check the detail level
        if sub_task_depth < 1:
            raise ValueError("The sub task depth must be greater than 0.")
        elif sub_task_depth > 5:
            raise ValueError("The sub task depth must be less than 5.")
        
        # Update the environment status to created
        self.to_created()
        
        # Create a new Task
        task = BaseTreeTaskNode(
            name=f"任务{len(self.tasks) + 1}",
            objective=question, 
            key_results=description, 
            sub_task_depth=sub_task_depth,
        )
        # Set the task as the sub-task
        self.tasks[task.name] = task
        # Log the task
        logger.info(f"任务创建: \n{task.objective}")
        
        # Process the task
        task = await self.process_task(task, completion_config=completion_config, **kwargs)
        # Get the blueprint of the task
        blueprint: str = self.context.get("blueprint")
        # Get the JSON view of the task
        todo_view = ToDoTaskView(task).format()
        # Get the JSON view of the task
        json_view = JsonTaskView(task).format()
        # Return the history of created task and the blueprint and the ToDo view of the task
        return (
            blueprint,  # blueprint
            todo_view,  # todo view
            json.dumps(json.loads(json_view)['任务1']['sub_tasks'], ensure_ascii=False, indent=4),  # json view
        )
        
    async def process_task(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        completion_config: BaseCompletionConfig = None,
        **kwargs,
    ) -> TreeTaskNode:
        """Process the task.
        
        Args:
            target (TreeTaskNode):
                The target task to be processed.
            max_error_retry (int, optional):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional):
                The maximum number of times to idle thinking the agent. 
            completion_config (BaseCompletionConfig, optional):
                The completion config of the environment.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        
        Returns:
            TreeTaskNode:
                The processed task.
        """
        # Initialize the error retry count
        current_error = 0
        
        while not target.is_finished():
            # Check the status of the target
            if self.is_created():
                # Call for global orchestration
                target = await self.orchestrate(
                    target=target, 
                    max_error_retry=max_error_retry, 
                    max_idle_thinking=max_idle_thinking, 
                    completion_config=completion_config,
                    **kwargs,
                )
            
            elif self.is_running():
                # Break the loop
                break
            
            elif self.is_error():
                # Get all the sub-tasks that are not finished
                sub_tasks = [sub_task for sub_task in target.sub_tasks.values() if not sub_task.is_finished()]
                # Delete all the sub-tasks that are not finished
                for sub_task in sub_tasks:
                    # Log the deletion
                    logger.info(f"删除子任务: {sub_task.name}: {sub_task.objective}")
                    # Delete the sub-task
                    del target.sub_tasks[sub_task.name]
                
                # Rollback the target to created status
                target.to_created()
                # Log the rollback
                logger.info(f"回滚到创建状态: {target.name}: {target.objective}")
                # Rollback the self to created status
                self.to_created()
                # Log the rollback
                logger.info(f"回滚到创建状态: {self.uid}")
                # Clean up the error information
                target.results = ""
            
            elif self.is_cancelled():
                # Increment the error retry count
                current_error += 1
                # Log the error
                logger.error(f"任务 {target.objective} 处理失败，重试次数: {current_error} / {max_error_retry}。")
                
                # Check the error retry count
                if current_error >= max_error_retry:
                    # Log the error
                    logger.error(f"任务 {target.objective} 处理失败，达到最大重试次数。")
                    # Raise the error
                    raise RuntimeError(f"Task {target.objective} is in error state. Max error retry count reached.")
                
                # Rollback the target to error status
                target.to_error()
                # Rollback the self to error status
                self.to_error()
                
            else:
                # Log the error
                logger.critical(f"任务 {target.objective} 当前处于非法状态 {target.get_status()}。")
                # Raise the error
                raise RuntimeError(f"Task {target.objective} is in error state. Invalid status.")
            
            # Check if the target is finished
            if target.is_finished():
                # Break the loop
                break
        
        # Return the answer
        return target
        
    async def orchestrate(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        completion_config: BaseCompletionConfig = None,
        **kwargs,
    ) -> TreeTaskNode:
        """Orchestrate the task. 
        
        Args:
            target (TreeTaskNode):
                The target task to be orchestrated.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of times to idle thinking the agent. 
            completion_config (BaseCompletionConfig, optional):
                The completion config of the environment.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        
        Returns:    
            TreeTaskNode:
                The orchestrated task.
        """
        # Call for global orchestration
        message: AssistantMessage = await self.call_agent(
            AgentType.ORCHESTRATE, 
            target=target, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config,
            **kwargs,
        )
        # Log the message
        logger.info(f"Agent Response: \n{message.content}")
        
        if not target.is_error(): 
            # Set self to running
            self.to_running()
        else:
            # Set self to error
            self.to_error()
        
        # Return the target
        return target
