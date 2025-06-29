import re
import inspect
from enum import Enum
from traceback import format_exc
from typing import Callable, Union, OrderedDict

from loguru import logger
from mcp import Tool as MCPTool
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.message import CompletionMessage, ToolCallRequest, ToolCallResult, MessageRole, StopReason
from myagents.core.interface import Agent, Workflow, Logger, Task, Environment, Context, EnvironmentStatus, TaskStatus
from myagents.core.envs.task import BaseTask, TaskAnswerView
from myagents.core.envs.base import BaseEnvironment
from myagents.core.utils.tools import ToolView
from myagents.tools.docs import DocumentLog, BaseDocument, FormatType
from myagents.prompts.envs.query import QUERY_SYSTEM_PROMPT, QUERY_POST_PROCESS_PROMPT


class OutputType(Enum):
    """
    The type of the output. 
    - "summary": The output should be a summary of the answer. Only the summary would be output to user. 
    - "document": The output should be a document of the answer. This means that you can only modify 
    the content of the answer by `diff` command. And all the content would be output to user. 
    """
    SUMMARY = "summary"
    DOCUMENT = "document"


class Query(BaseEnvironment):
    """Query is the environment for the query and answer the question.
    
    Attributes:
        system_prompt (str):
            The system prompt of the environment.
        
        agent (Agent):
            The agent that will be used to answer the question.
        debug (bool):
            The debug flag.
        custom_logger (Logger):
            The custom logger.
        tools (dict[str, Union[FastMcpTool, MCPTool]]):
            The tools of the environment.
        tool_functions (dict[str, Callable]):
            The functions of the tools.
        workflows (dict[str, Workflow]):
            The workflows of the environment. 
        
        sub_tasks (dict[str, Task]):
            The sub-tasks of the environment. The key is the sub-task name and the value is the sub-task.  
        history (list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]]):
            The history of the environment.
        answer (str):
            The answer to the question.
    """
    system_prompt: str = QUERY_SYSTEM_PROMPT
    
    agent: Agent
    debug: bool
    custom_logger: Logger
    context: Context
    
    tools: dict[str, Union[FastMcpTool, MCPTool]]
    tool_functions: dict[str, Callable]
    workflows: dict[str, Workflow]
    
    tasks: OrderedDict[str, Task]
    answers: OrderedDict[str, str]
    status: EnvironmentStatus
    history: list[Union[CompletionMessage, ToolCallRequest, ToolCallResult]]
    
    def __init__(
        self,
        agent: Agent, 
        custom_logger: Logger = logger, 
        debug: bool = False, 
        workflows: dict[str, Workflow] = {}, 
        *args: tuple, 
        **kwargs: dict, 
    ) -> None:
        """Initialize the Query environment.
        
        Args:
            agent (Agent):
                The agent that will be used to answer the question.
            custom_logger (Logger):
                The custom logger.
            debug (bool):
                The debug flag.
            workflows (dict[str, Workflow], optional):
                The workflows that will be orchestrated to process the task.
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        super().__init__(agent=agent, custom_logger=custom_logger, debug=debug, workflows=workflows, *args, **kwargs)
        
        # Check the rpa workflow is in the workflows
        if "rpa" not in self.workflows:
            raise ValueError(f"The `rpa` workflow is not in the workflows.")
        
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
            self.custom_logger.error(f"Query 注册的工具为空: {format_exc()}")
            raise RuntimeError("No tools registered for the query environment.")
        
        # Check the tools
        tool_str = ""
        for tool in self.tools.values():
            tool_str += f"{ToolView(tool).format()}\n"
        self.custom_logger.info(f"Tools: \n{tool_str}")

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
            message = CompletionMessage(
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
            self.history.append(message)
            # Call for completion
            message: CompletionMessage = await self.agent.think(self.history, allow_tools=True)
            # Record the completion message
            self.history.append(message)
                
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
                        self.custom_logger.error(f"工具调用 {tool_call.name} 失败: \n{tool_result.content}")
                    
                    # Update the messages
                    task.history[TaskStatus.RUNNING].append(tool_result)
                    
            elif finish_flag:
                self.custom_logger.info(f"总结或修订任务执行完成: \n{task.question}") 
                # Set the status to finished
                self.status = EnvironmentStatus.FINISHED
                break
            else:
                # Update the current thinking
                current_thinking += 1
            
            # Check if the current thinking is reached the limit
            if current_thinking >= max_thinking:
                # The current thinking is reached the limit, end the workflow
                self.custom_logger.error(f"当前任务执行已达到最大思考次数，总结或修订任务执行结束: \n{task.question}")
                self.status = EnvironmentStatus.FINISHED
                # Force the loop to break
                break
        
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
                
        Returns:
            str:
                The answer to the question. 
                
        Raises:
            ValueError:
                The detail level is not valid.
        """
        # Check the detail level
        if detail_level < 1:
            raise ValueError("The detail level must be greater than 0.")
        elif detail_level > 5:
            raise ValueError("The detail level must be less than 5.")
        
        # Set the status to running
        self.status = EnvironmentStatus.RUNNING
        # Record the question
        self.history.append(CompletionMessage(role=MessageRole.USER, content=question))
        # Create a new Task
        task = BaseTask(
            question=question, 
            description=description, 
            detail_level=detail_level,
        )
        # Set the task as the sub-task
        self.tasks[task.question] = task
        task.parent = self.tasks[task.question]
        # Log the task
        self.custom_logger.info(f"任务创建: \n{task.question}")
        # Run the react flow
        task = await self.workflows["rpa"].run(task)
        
        # Check the output type
        if output_type == OutputType.SUMMARY:
            # Set the answer view of the task as the output history
            # This is used for the summary output type. 
            self.history.append(
                CompletionMessage(
                    role=MessageRole.ASSISTANT, 
                    content=TaskAnswerView(task).format(),
                )
            )
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
            # Append as the assistant response
            self.history.append(
                CompletionMessage(
                    role=MessageRole.ASSISTANT, 
                    content=line_view,
                )
            )
            # Post process the answer
            # Set the document to the context
            self.context = self.context.create_next(document=document, task=task)
            # Post process the answer
            answer = await self.__post_process(task, output_type)
            # Record the answer
            self.answers[task.question] = answer
            # Return the answer
            return answer
        
        else:
            raise ValueError(f"Unknown output type: {output_type}")
    
    async def call_tool(self, ctx: Union[Task, Environment], tool_call: ToolCallRequest, **kwargs: dict) -> ToolCallResult:
        """Call a tool to modify the environment.
        """
        # Check the tool call name
        if tool_call.name == "diff_modify":
            # Modify the document
            await self.tool_functions["diff_modify"](**tool_call.args)
            return ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content="The document is modified.",
            )
        
        elif tool_call.name == "finish_query":
            # Finish the query
            await self.tool_functions["finish_query"]()
            return ToolCallResult(
                tool_call_id=tool_call.id, 
                is_error=False, 
                content="The query is finished.",
            )
        else:
            raise ValueError(f"Unknown tool call name: {tool_call.name}, query environment allow only diff_modify and finish_query.")
        
        # Done the current context
        self.context = self.context.done()
        
        # Return the result
        return ToolCallResult(tool_call_id=tool_call.id, content="No tool is available.", is_error=True)
    