import uuid
import inspect
from asyncio import Semaphore
from collections import OrderedDict
from enum import Enum
from traceback import format_exc
from typing import Union

from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.messages import AssistantMessage, UserMessage, SystemMessage, ToolCallResult
from myagents.core.interface import Agent, AgentType, TreeTaskNode, Context, TaskStatus, Stateful
from myagents.core.tasks.task import BaseTreeTaskNode, DocumentTaskView, ToDoTaskView
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
    
    def __init__(
        self, 
        profile: str = "", 
        system_prompt: str = "", 
        doc_prompt: str = "", 
        task_prompt: str = "", 
        select_prompt: str = "", 
        *args, 
        **kwargs,
    ) -> None:
        """Initialize the Query environment.
        
        Args:
            profile (str, optional, defaults to ""):
                The profile of the environment.
            system_prompt (str, optional, defaults to ""):
                The system prompt of the environment.
            doc_prompt (str, optional, defaults to ""):
                The document prompt of the environment.
            task_prompt (str, optional, defaults to ""):
                The task prompt of the environment.
            select_prompt (str, optional, defaults to ""):
                The select prompt of the environment.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        super().__init__(*args, **kwargs)
        
        # Initialize the basic information
        self.profile = profile if profile != "" else PROFILE
        # Update the prompts
        self.prompts.update({
            "system": system_prompt if system_prompt != "" else QUERY_SYSTEM_PROMPT,
            "doc": doc_prompt if doc_prompt != "" else QUERY_DOC_PROMPT,
            "task": task_prompt if task_prompt != "" else QUERY_TASK_PROMPT,
            "select": select_prompt if select_prompt != "" else QUERY_SELECT_PROMPT,
        })
        
        # Initialize the tasks
        self.tasks = OrderedDict()
        
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
            task: TreeTaskNode = self.context.get("task")
            # Get the tool call
            tool_call = self.context.get("tool_call")
            # Set the answer to self.answers
            self.answers[task.question] = answer
            # Set the status to finished
            self.to_finished()
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
        
        # Set the status to running
        self.to_running()
        # Append the system prompt to the history
        self.update(SystemMessage(content=self.prompts["system"].format(
            profile=self.profile, 
            question_type=output_type.value, 
        )))
        # Create a new Task
        task = BaseTreeTaskNode(
            question=question, 
            description=description, 
            sub_task_depth=sub_task_depth, 
        )
        # Set the task as the sub-task
        self.tasks[task.question] = task
        # Log the task
        logger.info(f"任务创建: \n{task.question}")
        # Record the question
        self.update(UserMessage(content=QUERY_TASK_PROMPT.format(task=ToDoTaskView(task).format())))
        # Call for global orchestration
        message: AssistantMessage = await self.call_agent(
            AgentType.ORCHESTRATE, 
            target=task, 
            tool_choice="create_task", 
        )
        # Update the environment history
        self.update(message)
        # Plan and execute the task
        task = await self.plan_and_exec(task)
        
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
            content = DocumentTaskView(task).format()
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
            content = DocumentTaskView(task).format()
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

    async def plan_and_exec(self, target: TreeTaskNode) -> TreeTaskNode:
        """Plan and execute the task.
        """
        sub_tasks = target.sub_tasks
        # Create a iterator for the sub-tasks
        sub_tasks_iter = iter(sub_tasks.values())
        # Get the first sub-task
        sub_task = next(sub_tasks_iter)
        
        while not target.is_finished():

            # Process the created status
            if sub_task.is_created():
                # Create a new UserMessage
                user_message = UserMessage(content=self.prompts["task"].format(task=ToDoTaskView(sub_task).format()))
                # Update the environment history
                self.update(user_message)
                # Call for plan and execute
                message: AssistantMessage = await self.call_agent(
                    AgentType.PLAN_AND_EXECUTE, 
                    target=sub_task, 
                    exclude_tools=[*self.tools.keys()], 
                )
                # Update the environment history
                self.update(message)
                
            elif sub_task.is_error():
                # Log the error
                logger.error(f"Sub-task {sub_task.question} is in error state.")
                # Raise the error
                raise RuntimeError(f"Sub-task {sub_task.question} is in error state.")
            
            elif sub_task.is_finished():
                # Get the next sub-task
                try:
                    sub_task = next(sub_tasks_iter)
                except StopIteration:
                    # Log the content
                    logger.info(f"所有子任务已处理完成。")
                    # Break the loop
                    break
        
        return target
