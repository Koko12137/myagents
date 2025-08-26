from fastmcp.settings import logger
from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import Agent, Workflow, Workspace, CallStack
from myagents.core.tools_mixin import ToolsMixin


class BaseWorkflow(Workflow, ToolsMixin):
    """BaseWorkflow is the base class for all the workflows.
    
    Attributes:
        tools (dict[str, FastMcpTool]):
            The tools of the workflow.
        workspace (Workspace):
            The workspace of the workflow.
        call_stack (CallStack):
            The call stack of the workflow.
        
        profile (str):
            The profile of the workflow.
        agent (Agent):
            The agent that is used to work with the workflow.
        prompts (dict[str, str]):
            The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
        observe_formats (dict[str, str]):
            The format of the observation. The key is the observation name and the value is the format content. 
        sub_workflows (dict[str, Workflow]):
            The sub-workflows of the workflow. The key is the name of the sub-workflow and the value is the 
            sub-workflow instance. 
    """
    
    # Tools
    tools: dict[str, FastMcpTool]
    # Workspace
    workspace: Workspace
    # Call stack information
    call_stack: CallStack
    # Basic information
    profile: str
    agent: Agent
    prompts: dict[str, str]
    observe_formats: dict[str, str]
    # Sub-worflows
    sub_workflows: dict[str, Workflow]
    
    def __init__(
        self, 
        call_stack: CallStack,
        workspace: Workspace,
        profile: str, 
        prompts: dict[str, str] = {}, 
        observe_formats: dict[str, str] = {}, 
        sub_workflows: dict[str, Workflow] = {}, 
        **kwargs,
    ) -> None:
        """Initialize the BaseWorkflow.

        Args:
            call_stack (CallStack):
                The call stack of the workflow.
            workspace (Workspace):
                The workspace of the workflow.
            profile (str):
                The profile of the workflow.
            prompts (dict[str, str], optional):
                The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
            observe_formats (dict[str, str], optional):
                The formats of the observation. The key is the observation name and the value is the format method name. 
            sub_workflows (dict[str, Workflow], optional):
                The sub-workflows of the workflow. The key is the name of the sub-workflow and the value is the 
                sub-workflow instance. 
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Initialize the parent class
        super().__init__(call_stack=call_stack, **kwargs)
                
        # Initialize the workflow components
        self.profile = profile
        self.prompts = prompts
        self.observe_formats = observe_formats
        self.sub_workflows = sub_workflows
        
        # Initialize the agent
        self.agent = None
        # Initialize the workspace
        self.workspace = workspace

        # Post initialize
        self.post_init()
            
    def register_agent(self, agent: Agent) -> None:
        """Register an agent to the workflow.
        
        Args:
            agent (Agent):
                The agent to register.
        """
        # Check if the agent is registered
        if self.agent is not None:
            return 
        
        # Register the agent to the workflow
        self.agent = agent
        # Register the agent to the sub-workflows
        for sub_workflow in self.sub_workflows.values():
            sub_workflow.register_agent(agent)
