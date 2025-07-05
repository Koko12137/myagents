import re
import uuid
import inspect
from asyncio import Semaphore
from collections import OrderedDict
from enum import Enum
from traceback import format_exc
from typing import Union

from loguru import logger
from fastmcp.tools import Tool as FastMCPTool

from myagents.core.messages.message import AssistantMessage, UserMessage, SystemMessage, ToolCallResult, MessageRole, StopReason
from myagents.core.interface import Agent, AgentType, Task, Environment, Context, EnvironmentStatus, TaskStatus
from myagents.core.envs.task import BaseTask, TaskAnswerView
from myagents.core.envs.base import BaseEnvironment
from myagents.core.utils.tools import ToolView
from myagents.tools.docs import DocumentLog, BaseDocument, FormatType
from myagents.prompts.envs.query import QUERY_SYSTEM_PROMPT, QUERY_POST_PROCESS_PROMPT


NAME = "基础多轮问答环境"


PROFILE = """
问答环境，用于回答问题。回复内容可以是以下的类型：
- 总结：总结当前任务的答案。
- 文档：修改当前任务的答案。
- 选择：选择当前任务的答案。
"""


REQUIRED_AGENTS = [AgentType.ORCHESTRATOR, AgentType.REACTOR, AgentType.PLAN_AND_EXECUTOR]


class OutputType(Enum):
    """
    The type of the output. 
    - "summary": The output should be a summary of the answer. Only the summary would be output to user. 
    - "document": The output should be a document of the answer. This means that you can only modify 
    the content of the answer by `diff` command. And all the content would be output to user. 
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
        async def finish_query() -> None:
            """
            完成当前任务。当你认为当前任务已经完成时，你可以选择以下任一方式来完成任务：
            
            - 调用这个工具来完成任务。
            - 在消息中设置完成标志为 True，并且不要调用这个工具。 
            
            Args:
                None
                
            Returns:
                None
            """
            # Set the query status to finished
            self.status = EnvironmentStatus.FINISHED
            
        @self.register_tool("diff_modify")
        async def diff_modify(action: str, line_num: int, content: str) -> None:
            """
            修改当前任务的答案。
            
            Args:
                action (str):
                    对于当前文档的修改操作。可以为 "replace", "insert", "delete"。
                line_num (int):
                    当前文档的行号。
                content (str):
                    当前文档的修改内容。
            """
            # 获取当前文档对象
            document: BaseDocument = self.context.get("document")
            if document is None:
                raise ValueError("No document found in context for diff_modify.")
            # 解析diff字符串为DocumentLog列表
            logs = []
            logs.append(DocumentLog(action=action, line=line_num, content=content))
            if logs:
                document.modify(logs)
            # 更新当前文档对象
            document.modify(logs)
            # 更新当前文档对象
            self.context["document"] = document
        
        # Check the registered tools count
        if len(self.tools) == 0:
            logger.error(f"Query 注册的工具为空: {format_exc()}")
            raise RuntimeError("No tools registered for the query environment.")
        
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
            # Set the answer to self.answers
            self.answers[task.question] = answer
            # Set the status to finished
            self.status = EnvironmentStatus.FINISHED
            # Return the answer
            return answer
        
        # Check the tools
        tool_str = ""
        for tool in self.tools.values():
            tool_str += f"{ToolView(tool).format()}\n"
        logger.debug(f"Tools: \n{tool_str}")

    async def __post_process(self, task: Task, output_type: OutputType) -> str:
        """Post process the answer.
        
        Args:
            task (Task):
                The task to be post processed.
            output_type (OutputType):
                The type of the output.
                
        Returns:
            str:
                The answer to the question.
        """
        # Designate the tool choice
        if output_type == OutputType.SELECTION:
            tool_choice = self.tools["select_answer"]
        else:
            tool_choice = "auto"
        
        # Additional tools information for tool calling
        tools_str = "\n".join([ToolView(tool).format() for tool in self.tools.values()])
        # Get the command explanation
        command_explanation = inspect.getdoc(OutputType)
        # Get the context view of the task
        context_view = await self.agent.observe(task)
        
        # This is used for no tool calling thinking limit.
        # If the agent is thinking more than max_thinking times, the loop will be finished.
        max_thinking = 3
        current_thinking = 0
        
        while self.status == EnvironmentStatus.RUNNING:
            # Create a message for the post process
            message = AssistantMessage(
                role=MessageRole.USER, 
                content=QUERY_POST_PROCESS_PROMPT.format(
                    output_type=output_type, 
                    task=context_view,
                    tools=tools_str,
                    command_explanation=command_explanation,
                ), 
                stop_reason=StopReason.NONE, 
            )
            # Record the post process message
            self.update(message)
            # Call for completion
            message: AssistantMessage = await self.agent.think(
                self.history, 
                allow_tools=False, 
                external_tools=self.tools, 
                tool_choice=tool_choice,
            )
            # Record the completion message
            self.update(message)
            # Log the message
            logger.info(f"模型回复: \n{message.content}")
                
            # Extract the finish flag
            finish_flag = re.search(r"<finish_flag>\s*\n(.*)\n\s*</finish_flag>", message.content, re.DOTALL)
            if finish_flag:
                # Extract the finish flag
                finish_flag = finish_flag.group(1)
                # Check if the finish flag is True
                if finish_flag == "True":
                    finish_flag = True
                else:
                    finish_flag = False
            else:
                finish_flag = False
                
            # Check the stop reason
            if message.stop_reason == StopReason.TOOL_CALL:
                # Reset the current thinking
                current_thinking = 0
                
                # Traverse all the tool calls
                for tool_call in message.tool_calls:
                    try:
                        # Call from the agent. 
                        # If there is any error caused by the tool call, the flag `is_error` will be set to True. 
                        # However, if there is any error caused by the MCP client connection, this should raise a RuntimeError. 
                        tool_result = await self.agent.call_tool(task, tool_call)
                        
                    except Exception as e:
                        # May be caused by the workflow tools. 
                        tool_result = ToolCallResult(
                            tool_call_id=tool_call.id, 
                            is_error=True, 
                            content=e
                        )
                        logger.error(f"工具调用 {tool_call.name} 失败: \n{tool_result.content}")
                    # Update the messages
                    self.update(tool_result)
                    
            elif finish_flag:
                logger.info(f"总结或修订任务执行完成: \n{task.question}") 
                # Set the status to finished
                self.status = EnvironmentStatus.FINISHED
                break
            else:
                # Update the current thinking
                current_thinking += 1
                # Log the current thinking
                logger.warning(f"模型没有执行动作,当前思考次数: {current_thinking}")
            
                # Check if the current thinking is reached the limit
                if current_thinking >= max_thinking:
                    # The current thinking is reached the limit, end the workflow
                    logger.error(f"当前任务执行已达到最大思考次数，总结或修订任务执行结束: \n{task.question}")
                    self.status = EnvironmentStatus.FINISHED
                    # Announce the idle thinking
                    message = AssistantMessage(
                        role=MessageRole.USER, 
                        content=f"你目前已经达到了最大思考次数，你将没有机会更新答案。",
                    )
                    self.update(message)
                    # Force the loop to break
                    break
                
                # Announce the idle thinking
                message = AssistantMessage(
                    role=MessageRole.USER, 
                    content=f"你目前已经思考了 {current_thinking} 次，最多思考 {max_thinking} 次后会被强制停止思考并退出循环，并且你将没有机会更新答案。",
                )
                self.update(message)
        
        # Set the answer
        document: BaseDocument = self.context.get("document")
        # Return the answer
        return document.format(FormatType.ARTICLE)
                
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
        # Record the question
        self.history.append(AssistantMessage(role=MessageRole.USER, content=question))
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
        
        # Call for OKR orchestration
        message: AssistantMessage = await self.leader.run(task)
        # Update the environment history
        self.update(message)
        
        # Call for react flow
        message: AssistantMessage = await self.leader.run(task)
        # Update the environment history
        self.update(message)
        # Run the react flow
        message: AssistantMessage = await self.leader.run(task)
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
            # Set the answer view of the task as the output history
            # This is used for the summary output type. 
            self.update(
                AssistantMessage(
                    role=MessageRole.ASSISTANT, 
                    content=TaskAnswerView(task).format(),
                )
            )
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
            self.update(
                AssistantMessage(
                    role=MessageRole.ASSISTANT, 
                    content=line_view,
                )
            )
            """ [[ ## Post process the answer ##]] """
            # Set the document to the context
            self.context = self.context.create_next(document=document, task=task)
            # Post process the answer
            answer = await self.__post_process(task, output_type)
            # Record the answer
            self.answers[task.question] = answer
            # Log the answer
            logger.info(f"最终文档: \n{answer}")
            # Resume the context
            self.context = self.context.done()
            # Return the answer
            return answer
        
        elif output_type == OutputType.SELECTION:
            # Set the answer view of the task as the output history
            # This is used for the selection output type. 
            content = TaskAnswerView(task).format()
            # Log the content
            logger.info(f"选择题分析过程: \n{content}")
            # Update the history
            self.update(
                AssistantMessage(
                    role=MessageRole.ASSISTANT, 
                    content=content,
                )
            )
            """ [[ ## Post process the answer ##]] """
            # Set the task to the context
            self.context = self.context.create_next(task=task)
            # Post process the answer
            answer = await self.__post_process(task, output_type)
            # Record the answer
            self.answers[task.question] = answer
            # Log the answer
            logger.info(f"最终答案: \n{answer}")
            # Resume the context
            self.context = self.context.done()
            # Return the answer
            return content
        
        else:
            raise ValueError(f"Unknown output type: {output_type}")
