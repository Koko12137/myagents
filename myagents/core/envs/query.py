from asyncio import Semaphore
from collections import OrderedDict
from enum import Enum
from traceback import format_exc
from typing import Union

from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.messages import AssistantMessage, UserMessage, SystemMessage, ToolCallResult
from myagents.core.interface import Agent, TreeTaskNode, Context
from myagents.core.agents import AgentType
from myagents.core.tasks import BaseTreeTaskNode, DocumentTaskView, ToDoTaskView
from myagents.core.envs.base import BaseEnvironment, EnvironmentStatus
from myagents.core.utils.tools import ToolView
from myagents.tools.docs import DocumentLog, BaseDocument, FormatType
from myagents.prompts.envs.query import (
    NAME, 
    PROFILE, 
    QUERY_SYSTEM_PROMPT, 
    QUERY_DOC_PROMPT, 
    QUERY_ORCHESTRATION_PROMPT, 
    QUERY_PLAN_AND_EXECUTE_PROMPT, 
    QUERY_SELECT_PROMPT, 
    QUERY_ERROR_PROMPT,
)


REQUIRED_AGENTS = [AgentType.ORCHESTRATE, AgentType.REACT, AgentType.PLAN_AND_EXECUTE]


class OutputType(Enum):
    """
    The type of the output. 
    - "summary": The output should be a summary of the answer. Only the summary would be output to user. 
    - "document": The output should be a document of the answer. This means that you can only modify 
    the content of the answer by `diff` command. And all the content would be output to user. 
    - "selection": The output should be a selection of the answer. This means that you can only select 
    the answer by `select_answer` command. 
    """
    SUMMARY = "summary"
    DOCUMENT = "document"
    SELECTION = "selection"


class Query(BaseEnvironment):
    """Query is the environment for the multi-turn query and answer the question. The answer type can be:
     - Summary: The summary of the answer.
     - Document: The document of the answer.
     - Selection: The selection of the answer.
    
    Attributes:
        uid (str):
            The unique identifier of the environment. 
        name (str):
            The name of the environment.
        profile (str):
            The profile of the environment. 
        system_prompt (str):
            The system prompt of the environment. 
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
        profile: str = "", 
        system_prompt: str = "", 
        orchestration_prompt: str = "", 
        plan_and_execute_prompt: str = "", 
        doc_prompt: str = "", 
        select_prompt: str = "", 
        error_prompt: str = "", 
        *args, 
        **kwargs,
    ) -> None:
        """Initialize the Query environment.
        
        Args:
            profile (str, optional, defaults to ""):
                The profile of the environment.
            system_prompt (str, optional, defaults to ""):
                The system prompt of the environment.
            orchestration_prompt (str, optional, defaults to ""):
                The orchestration prompt of the environment.
            doc_prompt (str, optional, defaults to ""):
                The document prompt of the environment.
            select_prompt (str, optional, defaults to ""):
                The select prompt of the environment.
            error_prompt (str, optional, defaults to ""):
                The error prompt of the environment.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        super().__init__(
            name=NAME, 
            profile=profile if profile != "" else PROFILE, 
            prompts={
                "system": system_prompt if system_prompt != "" else QUERY_SYSTEM_PROMPT,
                "orchestration": orchestration_prompt if orchestration_prompt != "" else QUERY_ORCHESTRATION_PROMPT,
                "plan_and_execute": plan_and_execute_prompt if plan_and_execute_prompt != "" else QUERY_PLAN_AND_EXECUTE_PROMPT,
                "doc": doc_prompt if doc_prompt != "" else QUERY_DOC_PROMPT,
                "select": select_prompt if select_prompt != "" else QUERY_SELECT_PROMPT,
                "error": error_prompt if error_prompt != "" else QUERY_ERROR_PROMPT,
            }, 
            required_agents=REQUIRED_AGENTS, 
            *args, 
            **kwargs,
        )
        
        # Initialize the tasks
        self.tasks = OrderedDict()
        # Initialize the answers
        self.answers = OrderedDict()
        # Post initialize
        self.post_init()
        
    def post_init(self) -> None:
        """Post init is the method that will be called after the initialization of the workflow.
        """
        # Initialize the tools of parent class
        super().post_init()
            
        @self.register_tool("diff_modify")
        async def diff_modify(action: str, line_num: int, content: str) -> ToolCallResult:
            """
            修改当前任务的答案。
            
            Args:
                action (str):
                    对于当前文档的修改操作。可以为 "replace", "insert", "delete"。
                line_num (int):
                    当前文档的行号。
                content (str):
                    当前文档的修改内容。
                
            Returns:
                ToolCallResult:
                    The tool call result.
            """
            # Get the tool call
            tool_call = self.context.get("tool_call")
            # 获取当前文档对象
            document: BaseDocument = self.context.get("document")
            # 解析diff字符串为DocumentLog列表
            logs = []
            logs.append(DocumentLog(action=action, line=line_num, content=content))
            if logs:
                document.modify(logs)
            # 更新当前文档对象
            document.modify(logs)
            # 更新当前文档对象
            self.context["document"] = document
            
            # Create a new tool call result
            result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content=f"文档已修改。",
            )
            return result
        
        @self.register_tool("select_answer")
        async def select_answer(answer: str) -> str:
            """
            选择当前任务的答案。如果是选择题，则需要调用这个工具来选择答案，你传入的选项必须是题目中给出的选项。
            
            Args:
                answer (str):
                    The answer to the question. 
                    - 如果是选择题，则需要传入题目中给出的选项，例如："A"。注意不要有任何其他字符。
                    - 如果是填空题，则需要传入填空的内容。
                    【注意】：其他类型题目，请不要调用这个工具。
            Returns:
                str:
                    选项（不允许包含除选项外的任何字符）或者填空的内容。
            """
            # Get the task
            target: TreeTaskNode = self.context.get("target")
            # Get the tool call
            tool_call = self.context.get("tool_call")
            # Set the answer to self.answers
            self.answers[target.tasks[f"任务{len(self.tasks)}"].uid] = answer
            # Create a new tool call result
            result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content=f"答案已选择。",
            )
            return result
        
        # Check the registered tools count
        if len(self.tools) == 0:
            logger.error(f"Query 注册的工具为空: {format_exc()}")
            raise RuntimeError("No tools registered for the query environment.")
        
        # Check the tools
        tool_str = ""
        for tool in self.tools.values():
            tool_str += f"{ToolView(tool).format()}\n"
        logger.debug(f"Tools: \n{tool_str}")
                
    async def run(
        self, 
        question: str, 
        description: str, 
        sub_task_depth: int = 3, 
        output_type: OutputType = OutputType.SUMMARY,
    ) -> str:
        """Run the query and answer the question.
        
        Args:
            question (str):
                The question to be answered.
            description (str):
                The detail information and limitation of the task.
            sub_task_depth (int):
                The max number of layers of sub-question layers that can be split from the question. 
                The sub task depth should be greater than 0 and less than 5.
            output_type (OutputType):
                The type of the output. 
                - OutputType.SUMMARY: The summary of the answer.
                - OutputType.DOCUMENT: The document of the answer.
                - OutputType.SELECTION: The selection of the answer.
        Returns:
            str:
                The answer to the question. 
                
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
        
        # Append the system prompt to the history
        self.update(SystemMessage(
            content=self.prompts["system"].format(
                profile=self.profile, 
                question_type=output_type.value, 
            ),
        ))
        # Create a new Task
        task = BaseTreeTaskNode(
            uid=f"任务{len(self.tasks) + 1}",
            objective=question, 
            key_results=description, 
            sub_task_depth=sub_task_depth,
        )
        # Set the task as the sub-task
        self.tasks[task.uid] = task
        # Log the task
        logger.info(f"任务创建: \n{task.objective}")
        
        # Process the task
        task = await self.process_task(task)
        
        # Check the task status
        if not task.is_finished():
            # Log the error
            logger.critical(f"Task {task.objective} is not finished.")
            # Raise the error
            raise RuntimeError(f"Task {task.objective} is not finished.")
        
        # Check the output type
        if output_type == OutputType.SUMMARY:
            # Log the content
            logger.info(f"最终答案: \n{task.outputs}")
            # Record the answer
            self.answers[task.uid] = task.outputs
            # Return the answer
            return task.outputs
        
        elif output_type == OutputType.DOCUMENT:
            # Set the answer view of the task as the output history
            # This is used for the document output type. 
            content = DocumentTaskView(task).format()
            # Create a new document
            document = BaseDocument(original_content=content)
            # Format to line view
            line_view = document.format(FormatType.LINE)
            # Log the content
            logger.info(f"文档按[行号-内容]输出: \n{line_view}")
            
            """ [[ ## Post process the answer ## ]] """
            # Call for react agent to modify the document or select the answer
            message: AssistantMessage = await self.call_agent(
                AgentType.REACT, 
                target=self, 
                document=document, 
                completion_config={
                    "exclude_tools": ["select_answer"],
                },
            )
            # Update the environment history
            self.update(message)
            # Log the answer
            logger.info(f"Agent Response: \n{message.content}")
            # Return the answer
            return message.content
        
        elif output_type == OutputType.SELECTION:
            # Call for react agent to select the answer
            message: AssistantMessage = await self.call_agent(
                AgentType.REACT, 
                target=self, 
                completion_config={
                    "tool_choice": "select_answer",
                },
            )
            # Set the status to finished
            self.to_finished()
            # Log the answer
            logger.info(f"Agent Response: \n{message.content}")
            # Return the answer
            return self.answers[task.uid]
        
        else:
            raise ValueError(f"Unknown output type: {output_type}")
        
    async def process_task(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
    ) -> TreeTaskNode:
        """Process the task.
        
        Args:
            target (TreeTaskNode):
                The target task to be processed.
            max_error_retry (int, optional):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional):
                The maximum number of times to idle thinking the agent. 
        
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
                )
            
            elif self.is_running():
                # Plan and execute the task
                target = await self.plan_and_exec(
                    target=target, 
                    max_error_retry=max_error_retry, 
                    max_idle_thinking=max_idle_thinking, 
                )
            
            elif self.is_error():
                # Get all the sub-tasks that are not finished
                sub_tasks = [sub_task for sub_task in target.sub_tasks.values() if not sub_task.is_finished()]
                # Delete all the sub-tasks that are not finished
                for sub_task in sub_tasks:
                    del target.sub_tasks[sub_task.uid]
                
                # Rollback the target to created status
                target.to_created()
                # Rollback the self to created status
                self.to_created()
                # Record the error information
                current_result = DocumentTaskView(target).format()
                # Create a new user message to record the error and the current result
                message = UserMessage(content=self.prompts["error"].format(
                    error_retry=current_error, 
                    max_error_retry=max_error_retry, 
                    error_reason=target.results,
                    current_result=current_result,
                ))  # BUG: 这里要先记录错误状态，再提示重试
                # Update the environment history
                self.update(message)
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
    ) -> TreeTaskNode:
        """Orchestrate the task. 
        
        Args:
            target (TreeTaskNode):
                The target task to be orchestrated.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of times to idle thinking the agent. 
        
        Returns:    
            TreeTaskNode:
                The orchestrated task.
        """
        # Prepare the completion config
        completion_config = {
            "exclude_tools": ["select_answer", "diff_modify"],
        }
        
        # Record the question
        message = UserMessage(
            content=self.prompts["orchestration"].format(task=ToDoTaskView(target).format()),
        )
        # Update the environment history
        self.update(message)
        # Log the message
        logger.info(f"Observe: \n{message.content}")
        # Call for global orchestration
        message: AssistantMessage = await self.call_agent(
            AgentType.ORCHESTRATE, 
            target=target, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config,
        )
        # Update the environment history
        self.update(message)
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

    async def plan_and_exec(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 1, 
    ) -> TreeTaskNode:
        """Plan and execute the task.
        
        Args:
            target (TreeTaskNode):
                The target task to be planned and executed.
            max_error_retry (int, optional, defaults to 3):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int, optional, defaults to 1):
                The maximum number of times to idle thinking the agent. 
        
        Returns:
            TreeTaskNode:
                The planned and executed task.
        """
        # Prepare the completion config
        completion_config = {
            "exclude_tools": ["select_answer", "diff_modify"],
        }
        
        # Record the question
        message = UserMessage(
            content=self.prompts["plan_and_execute"].format(task=ToDoTaskView(target).format()),
        )
        # Update the environment history
        self.update(message)
        # Log the message
        logger.info(f"Observe: \n{message.content}")
        
        sub_tasks = target.sub_tasks
        # Create a iterator for the sub-tasks
        sub_tasks_iter = iter(sub_tasks.values())
        # Get the first sub-task
        sub_task = next(sub_tasks_iter)
        
        while not self.is_finished():

            # Process the created status
            if sub_task.is_created() or sub_task.is_running():
                # Create a new user message to record the question
                message = UserMessage(
                    content=self.prompts["plan_and_execute"].format(task=ToDoTaskView(sub_task).format()),
                )
                # Update the environment history
                self.update(message)
                # Log the message
                logger.info(f"Observe: \n{message.content}")
                
                # Call for plan and execute
                message: AssistantMessage = await self.call_agent(
                    AgentType.PLAN_AND_EXECUTE, 
                    target=sub_task, 
                    max_error_retry=max_error_retry, 
                    max_idle_thinking=max_idle_thinking, 
                    completion_config=completion_config,
                )
                # Update the environment history
                self.update(message)
                # Log the message
                logger.info(f"Agent Response: \n{message.content}")
                
            elif sub_task.is_cancelled():
                # Get all the sub-tasks that are not finished
                sub_tasks = [t for t in target.sub_tasks.values() if not t.is_finished()]
                # Delete all the sub-tasks that are not finished
                for t in sub_tasks:
                    del target.sub_tasks[t.uid]
                # Rollback the target to created status
                target.to_created()
                # Rollback the self to created status
                self.to_created()
                # Log the error
                logger.error(f"任务 {target.objective} 的所有未完成子任务已取消。")
                # Break the loop
                return target
            
            elif sub_task.is_finished():
                # Get the next sub-task
                try:
                    sub_task = next(sub_tasks_iter)
                except StopIteration:
                    # Log the content
                    logger.info(f"所有子任务已处理完成。")
                    # Break the loop
                    break
            
            else:
                raise RuntimeError(f"Task {target.objective} is in error state. Invalid status.")
        
        # Check all the sub-tasks are finished
        if not all(sub_task.is_finished() for sub_task in sub_tasks.values()):
            # Get all the sub-tasks that are not finished
            sub_tasks = [t for t in target.sub_tasks.values() if not t.is_finished()]
            # Delete all the sub-tasks that are not finished
            for t in sub_tasks:
                del target.sub_tasks[t.uid]
            # Rollback the target to created status
            target.to_created()
            # Rollback the self to created status
            self.to_created()
            # Log the error
            logger.error(f"任务 {target.objective} 的所有未完成子任务已取消。")
            # Return the target
            return target
        
        # Create a new user message to record the question
        message = UserMessage(
            content=self.prompts["plan_and_execute"].format(task=ToDoTaskView(target).format()),
        )
        # Update the environment history
        self.update(message)
        # Log the message
        logger.info(f"Observe: \n{message.content}")
        # Call for plan and execute
        message: AssistantMessage = await self.call_agent(
            AgentType.PLAN_AND_EXECUTE, 
            target=target, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config, 
        )
        # Update the environment history
        self.update(message)
        # Log the message
        logger.info(f"Agent Response: \n{message.content}")
        return target

    async def observe(self, format: str = "document") -> str:
        """Observe the environment. The format can be:
         - "document": The document of the environment.
         - "summary": The summary of the environment.
         - "todo": The todo list of the environment.
         
        Returns:
            str:
                The observation of the environment.
        """
        # Get the task from the tasks
        task = list(self.tasks.values())[-1]
        # Return the document of the task
        if format == "document":
            return DocumentTaskView(task).format()
        elif format == "todo":
            return ToDoTaskView(task).format()
        else:
            raise ValueError(f"Unknown format: {format}")
