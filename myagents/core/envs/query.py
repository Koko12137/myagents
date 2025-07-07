import uuid
import inspect
from asyncio import Semaphore
from collections import OrderedDict
from enum import Enum
from traceback import format_exc
from typing import Union

from loguru import logger
from fastmcp.tools import Tool as FastMCPTool

from myagents.core.messages import AssistantMessage, UserMessage, SystemMessage, ToolCallResult
from myagents.core.interface import Agent, AgentType, Task, Context, TaskStatus
from myagents.core.envs.task import BaseTask, TaskAnswerView, TaskContextView
from myagents.core.envs.base import BaseEnvironment, EnvironmentStatus
from myagents.core.utils.tools import ToolView
from myagents.tools.docs import DocumentLog, BaseDocument, FormatType
from myagents.prompts.envs.query import QUERY_SYSTEM_PROMPT, QUERY_DOC_PROMPT, QUERY_TASK_PROMPT, QUERY_SELECT_PROMPT, NAME, PROFILE


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
        tools (dict[str, FastMCPTool]):
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
    uid: str = uuid.uuid4().hex
    name: str = NAME
    profile: str = PROFILE
    system_prompt: str = QUERY_SYSTEM_PROMPT
    required_agents: list[AgentType] = REQUIRED_AGENTS
    # Core components
    leader: Agent
    agents: dict[str, Agent]
    agent_type_map: dict[AgentType, list[str]]
    agent_type_semaphore: dict[AgentType, Semaphore]
    # Tools
    tools: dict[str, FastMCPTool]
    # Context
    context: Context
    # Status and history
    status: EnvironmentStatus
    history: list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallResult]]
    tasks: OrderedDict[str, Task]
    
    def __init__(self, *args, **kwargs) -> None:
        """Initialize the Query environment.
        
        Args:
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        super().__init__(*args, **kwargs)
        
        # Initialize the tasks
        self.tasks = OrderedDict()
        
        # Post initialize
        self.post_init()
        
    def post_init(self) -> None:
        """Post init is the method that will be called after the initialization of the workflow.
        """
        @self.register_tool("finish_query")
        async def finish_query() -> ToolCallResult:
            """
            完成当前任务。当你认为当前任务已经完成时，你可以选择以下任一方式来完成任务：
            
            - 调用这个工具来完成任务。
            - 在消息中设置完成标志为 True，并且不要调用这个工具。 
            
            Args:
                None
                
            Returns:
                ToolCallResult:
                    The tool call result.
            """
            # Get the tool call
            tool_call = self.context.get("tool_call")
            # Set the query status to finished
            self.status = EnvironmentStatus.FINISHED
            # Create a new tool call result
            result = ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content=f"任务已设置为 {EnvironmentStatus.FINISHED.value} 状态。",
            )
            return result
            
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
            task: Task = self.context.get("task")
            # Get the tool call
            tool_call = self.context.get("tool_call")
            # Set the answer to self.answers
            self.answers[task.question] = answer
            # Set the status to finished
            self.status = EnvironmentStatus.FINISHED
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
        detail_level: int = 3, 
        output_type: OutputType = OutputType.SUMMARY,
    ) -> str:
        """Run the query and answer the question.
        
        Args:
            question (str):
                The question to be answered.
            description (str):
                The detail information and limitation of the task.
            detail_level (int):
                The max number of layers of sub-question layers that can be split from the question. 
                The detail level should be greater than 0 and less than 5.
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
        if detail_level < 1:
            raise ValueError("The detail level must be greater than 0.")
        elif detail_level > 5:
            raise ValueError("The detail level must be less than 5.")
        
        # Set the status to running
        self.status = EnvironmentStatus.RUNNING
        # Append the system prompt to the history
        self.update(SystemMessage(content=self.system_prompt.format(
            profile=self.profile, 
            question_type=output_type.value, 
        )))
        # Create a new Task
        task = BaseTask(
            question=question, 
            description=description, 
            detail_level=detail_level, 
        )
        # Set the task as the sub-task
        self.tasks[task.question] = task
        # Log the task
        logger.info(f"任务创建: \n{task.question}")
        # Record the question
        self.update(UserMessage(content=QUERY_TASK_PROMPT.format(task=TaskContextView(task).format())))
        # Call for global orchestration
        message: AssistantMessage = await self.call_agent(AgentType.ORCHESTRATE, target=task)
        # Update the environment history
        self.update(message)
        # Create a new UserMessage
        user_message = UserMessage(content=QUERY_TASK_PROMPT.format(task=TaskContextView(task).format()))
        # Update the environment history
        self.update(user_message)
        # Call for plan and execute
        message: AssistantMessage = await self.call_agent(AgentType.PLAN_AND_EXECUTE, target=task)
        # Update the environment history
        self.update(message)
        
        # Check the task status
        if task.status != TaskStatus.FINISHED:
            # Log the error
            logger.critical(f"Task {task.question} is not finished.")
            # Raise the error
            raise RuntimeError(f"Task {task.question} is not finished.")
        
        # Check the output type
        if output_type == OutputType.SUMMARY:
            # Log the content
            logger.info(f"最终答案: \n{task.answer}")
            # Record the answer
            self.answers[task.question] = task.answer
            # Return the answer
            return task.answer
        
        elif output_type == OutputType.DOCUMENT:
            # Set the answer view of the task as the output history
            # This is used for the document output type. 
            content = TaskAnswerView(task).format()
            # Create a new document
            document = BaseDocument(original_content=content)
            # Format to line view
            line_view = document.format(FormatType.LINE)
            # Log the content
            logger.info(f"文档按[行号-内容]输出: \n{line_view}")
            # Append as the assistant response
            self.update(UserMessage(content=QUERY_DOC_PROMPT.format(task_lines=line_view)))
            
            """ [[ ## Post process the answer ## ]] """
            # Call for react agent to modify the document or select the answer
            message: AssistantMessage = await self.call_agent(
                AgentType.REACT, 
                target=task, 
                document=document, 
                exclude_tools=["select_answer"], 
            )
            # Update the environment history
            self.update(message)
            # Log the answer
            logger.info(f"模型回复: \n{message.content}")
            # Return the answer
            return message.content
        
        elif output_type == OutputType.SELECTION:
            # Set the answer view of the task as the output history
            # This is used for the selection output type. 
            content = TaskAnswerView(task).format()
            # Create a new UserMessage
            user_message = UserMessage(content=QUERY_SELECT_PROMPT.format(task=content))
            # Update the environment history
            self.update(user_message)
            # Call for react agent to select the answer
            message: AssistantMessage = await self.call_agent(
                AgentType.REACT, 
                target=task, 
                tool_choice="select_answer", 
            )
            # Update the environment history
            self.update(message)
            # Log the answer
            logger.info(f"模型回复: \n{message.content}")
            # Return the answer
            return message.content
        
        else:
            raise ValueError(f"Unknown output type: {output_type}")
