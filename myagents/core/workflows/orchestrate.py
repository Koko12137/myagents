from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import Agent, TreeTaskNode, TaskStatus, Stateful, Context
from myagents.core.messages import AssistantMessage, UserMessage, SystemMessage, ToolCallResult
from myagents.core.workflows import BaseWorkflow
from myagents.core.tasks import BaseTreeTaskNode
from myagents.core.utils.extractor import extract_by_label
from myagents.core.utils.strings import normalize_string
from myagents.prompts.workflows.orchestrate import PROFILE, SYSTEM_PROMPT, THINK_PROMPT, ACTION_PROMPT


class OrchestrateFlow(BaseWorkflow):
    """This is use for Orchestrating the task. This workflow will not design any detailed plans, it will 
    only orchestrate the key objectives of the task. 
        
    Attributes:
        profile (str):
            The profile of the workflow.
        system_prompt (str):
            The system prompt of the workflow. This is used to set the system prompt of the workflow. 
        agent (Agent): 
            The agent that is used to orchestrate the task.
        context (Context):
            The global context container of the workflow.
        tools (dict[str, FastMcpTool]):
            The tools can be used for the agent. 
    """
    # Basic information
    profile: str
    system_prompt: str
    agent: Agent
    # Context and tools
    context: Context
    tools: dict[str, FastMcpTool]

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the ReActFlow.

        Args:
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        super().__init__(*args, **kwargs)
        
        # Initialize the workflow components
        self.profile = PROFILE
        self.system_prompt = SYSTEM_PROMPT.format(profile=self.profile)
        self.agent = None
        
    async def post_init(self) -> None:
        """Post init is the method that will be called after the initialization of the workflow.
        
        This method will be called after the initialization of the workflow.
        """
        # Call the parent post init method
        await super().post_init()
        
        # Register the create task tool
        @self.register_tool("create_task")
        async def create_task(question: str, description: str) -> ToolCallResult:
            """
            创建一个新的任务，并将其添加到当前任务的子任务中。
            
            Args:
                question (str): 
                    当前任务需要回答或解决的问题。
                description (str): 
                    你需要用一段文字来描述当前问题应该达到什么目标，不能仅用一句话来描述。
                
            Returns:
                ToolCallResult: 
                    创建子任务的工具调用结果。
            """
            # Get the parent task from the context
            parent = self.context.get("task")
            # Get the function call details
            tool_call = self.context.get("tool_call")
            
            # Normalize the question and description
            question = normalize_string(question)
            description = normalize_string(description)
            
            # Create a new task
            new_task = BaseTreeTaskNode(
                question=question, 
                description=description, 
                sub_task_depth=parent.sub_task_depth - 1, 
            )
            
            # Check if the new task is a leaf task
            if new_task.sub_task_depth > 0:
                # Set the status of the new task
                new_task.to_created()
            else:
                # Set the status of the new task as running
                new_task.to_running()
                
            # Add the new task to the task
            parent.sub_tasks[new_task.question] = new_task
            # Add the parent task to the new task
            new_task.parent = parent
            
            # Create a new tool call result
            tool_call_result = ToolCallResult(
                tool_call_id=tool_call.id,
                content="new task created", 
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
        # Observe the task
        observe = await self.agent.observe(target)
        # Log the observation
        logger.info(f"Observe: \n{observe}")
        # Create a new message for the current observation
        message = UserMessage(content=THINK_PROMPT.format(observe=observe))
        # Append the reason prompt to the task history
        target.update(message)
    
        while target.is_created():
            # Call for completion
            message: AssistantMessage = await self.agent.think(target.get_history())
            # Log the message
            logger.info(f"Assistant Message: \n{message.content}")
            # Record the completion message
            target.update(message)
            
            # Extract the orchestration blueprint from the task by regular expression
            blueprint = extract_by_label(message.content, "orchestration")
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
    
    async def __act(
        self, 
        target: Stateful, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [], 
        *args, 
        **kwargs,
    ) -> Stateful:
        """Act about the task. This is the post step of the planning in order to execute the detailed 
        plan of the task. 
        
        Args:
            target (Stateful):
                The task to act about.
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
                The target after acting.
        """
        pass
        
    async def run(
        self, 
        target: Stateful, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
        tool_choice: str = None, 
        exclude_tools: list[str] = [], 
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
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.

        Returns:
            Stateful:
                The target after orchestrating.
        """
        # Update system prompt to history
        message = SystemMessage(content=self.system_prompt)
        target.update(message)
        
        # Reason about the task
        await self.__reason(
            target, 
            max_error_retry, 
            max_idle_thinking, 
            tool_choice, 
            exclude_tools, 
            *args, 
            **kwargs,
        )
    
        # Act about the task
        await self.__act(
            target, 
            max_error_retry, 
            max_idle_thinking, 
            tool_choice, 
            exclude_tools, 
            *args, 
            **kwargs,
        )
        
        return target
    