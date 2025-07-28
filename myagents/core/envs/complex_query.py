from asyncio import Semaphore
from collections import OrderedDict
from enum import Enum
from typing import Union

from loguru import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.messages import AssistantMessage, UserMessage, SystemMessage, ToolCallResult
from myagents.core.interface import Agent, TreeTaskNode, Context
from myagents.core.agents import AgentType
from myagents.core.tasks import BaseTreeTaskNode, DocumentTaskView, ToDoTaskView
from myagents.core.llms.config import BaseCompletionConfig
from myagents.core.envs.base import EnvironmentStatus
from myagents.core.envs.plan_and_exec import PlanAndExecEnv
from myagents.tools.docs import DocumentLog, BaseDocument, FormatType
from myagents.prompts.envs.query import (
    NAME, 
    PROFILE, 
    QUERY_SYSTEM_PROMPT, 
    QUERY_DOC_PROMPT, 
    QUERY_ORCHESTRATE_PROMPT, 
    QUERY_EXECUTE_PROMPT, 
    QUERY_SELECT_PROMPT, 
    QUERY_ERROR_PROMPT,
)


REQUIRED_AGENTS = [AgentType.ORCHESTRATE, AgentType.TREE_REACT]


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


class ComplexQuery(PlanAndExecEnv):
    """ComplexQuery is the environment for the multi-turn query and answer the question automatically. This environment 
    will split the question into atomic tasks by the orchestrate agents and then execute the tasks by the react agents. 
    The answer type can be:
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
        system_prompt: str = QUERY_SYSTEM_PROMPT, 
        orchestrate_prompt: str = QUERY_ORCHESTRATE_PROMPT, 
        execute_prompt: str = QUERY_EXECUTE_PROMPT, 
        doc_prompt: str = QUERY_DOC_PROMPT, 
        select_prompt: str = QUERY_SELECT_PROMPT, 
        error_prompt: str = QUERY_ERROR_PROMPT, 
        need_user_check: bool = False, 
        intent_recognition: bool = False, 
        **kwargs,
    ) -> None:
        """Initialize the Query environment.
        
        Args:
            profile (str):
                The profile of the environment.
            system_prompt (str):
                The system prompt of the environment.
            orchestrate_prompt (str):
                The orchestrate prompt of the environment.
            execute_prompt (str):
                The execute prompt of the environment.
            error_prompt (str):
                The error prompt of the environment.
            doc_prompt (str):
                The document prompt of the environment.
            select_prompt (str):
                The select prompt of the environment.
            need_user_check (bool, optional, defaults to False):
                Whether to need the user to check the orchestration blueprint.
            intent_recognition (bool, optional, defaults to False):
                Whether to need the user to recognize the intent of the question.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        super().__init__(
            name=NAME, 
            profile=profile, 
            prompts={
                "system": system_prompt,
                "orchestrate_prompt": orchestrate_prompt,
                "execute_prompt": execute_prompt,
                "error_prompt": error_prompt,
                "doc": doc_prompt,
                "select": select_prompt,
            }, 
            required_agents=REQUIRED_AGENTS, 
            **kwargs,
        )
        
        # Initialize the tasks
        self.tasks = OrderedDict()
        # Initialize the answers
        self.answers = OrderedDict()
        # Initialize the need user check
        self.need_user_check = need_user_check
        if self.need_user_check:
            self.required_agents.append(AgentType.PROXY)
        # Initialize the intent recognition
        self.intent_recognition = intent_recognition
        # Post initialize
        self.post_init()
    
    def __str__(self) -> str:
        """Get the string representation of the environment.
        """
        return f"ComplexQuery(profile={self.profile}, status={self.status})"
    
    def __repr__(self) -> str:
        """Get the string representation of the environment.
        """
        return self.__str__()
        
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
        async def select_answer(answer: str) -> ToolCallResult:
            """
            选择当前任务的答案。如果是选择题，则需要调用这个工具来选择答案，你传入的选项必须是题目中给出的选项。
            
            Args:
                answer (str):
                    The answer to the question. 
                    - 如果是选择题，则需要传入题目中给出的选项，例如："A"。注意不要有任何其他字符。
                    - 如果是填空题，则需要传入填空的内容。
                    【注意】：其他类型题目，请不要调用这个工具。
            
            Returns:
                ToolCallResult:
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
                
    async def run(
        self, 
        question: str, 
        description: str, 
        sub_task_depth: int = 3, 
        output_type: OutputType = OutputType.SUMMARY,
        completion_config: BaseCompletionConfig = None, 
        **kwargs,
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
            completion_config (BaseCompletionConfig, optional):
                The completion config of the environment.
            **kwargs:
                The keyword arguments to be passed to the parent class.
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
        task = await self.schedule(
            target=task, 
            completion_config=completion_config, 
            **kwargs,
        )
        
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
            self.answers[task.name] = task.outputs
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
            
            # Prepare the completion config
            if completion_config is None:
                completion_config = BaseCompletionConfig(
                    tool_choice="diff_modify",
                )
            else:
                # Update the completion config
                completion_config.update(tool_choice="diff_modify")
            
            """ [[ ## Post process the answer ## ]] """
            # Call for react agent to modify the document or select the answer
            message: AssistantMessage = await self.call_agent(
                AgentType.TREE_REACT, 
                target=self, 
                document=document, 
                completion_config=completion_config,
            )
            # Update the environment history
            self.update(message)
            # Log the answer
            logger.info(f"Agent Response: \n{message.content}")
            # Return the answer
            return message.content
        
        elif output_type == OutputType.SELECTION:
            # Prepare the completion config
            if completion_config is None:
                completion_config = BaseCompletionConfig(
                    tool_choice="select_answer",
                )
            else:
                # Update the completion config
                completion_config.update(tool_choice="select_answer")
            # Call for react agent to select the answer
            message: AssistantMessage = await self.call_agent(
                AgentType.TREE_REACT, 
                target=self, 
                completion_config=completion_config,
            )
            # Set the status to finished
            self.to_finished()
            # Log the answer
            logger.info(f"Agent Response: \n{message.content}")
            # Return the answer
            return self.answers[task.name]
        
        else:
            raise ValueError(f"Unknown output type: {output_type}")

    async def schedule(
        self, 
        target: TreeTaskNode, 
        max_error_retry: int = 3, 
        max_idle_thinking: int = 2, 
        completion_config: BaseCompletionConfig = None, 
        **kwargs, 
    ) -> TreeTaskNode:
        """Schedule the task.
        
        Args:
            target (TreeTaskNode):
                The target task to be scheduled.
            max_error_retry (int):
                The maximum number of times to retry the agent when the target is errored.
            max_idle_thinking (int):
                The maximum number of times to idle thinking the agent. 
            completion_config (BaseCompletionConfig):
                The completion config of the environment.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        
        Returns:
            TreeTaskNode:
                The scheduled task.
        """
        # Prepare the completion config
        if completion_config is None:
            completion_config = BaseCompletionConfig(
                exclude_tools=["select_answer", "diff_modify"],
            )
        else:
            # Update the completion config
            completion_config.update(
                exclude_tools=["select_answer", "diff_modify"],
            )
        
        # Call the ReAct Agent to reason about the real intent of the question
        # message: AssistantMessage = await self.call_agent(
        #     AgentType.REACT, 
        #     target=self, 
        #     completion_config=completion_config,
        # )
        # Update the environment history
        # self.update(message)
        # # Log the answer
        # logger.info(f"Agent Response: \n{message.content}")
        
        return await super().schedule(
            target=target, 
            max_error_retry=max_error_retry, 
            max_idle_thinking=max_idle_thinking, 
            completion_config=completion_config,
            **kwargs,
        )

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
