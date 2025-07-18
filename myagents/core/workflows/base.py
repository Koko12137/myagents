from fastmcp.tools import Tool as FastMcpTool

from myagents.core.interface import Agent, Workflow
from myagents.core.utils.context import BaseContext
from myagents.core.tools_mixin import ToolsMixin


class BaseWorkflow(Workflow, ToolsMixin):
    """BaseWorkflow is the base class for all the workflows.
    
    Attributes:
        context (BaseContext):
            The context of the workflow.
        tools (dict[str, FastMcpTool]):
            The tools of the workflow.
        
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
    # Context and tools
    context: BaseContext
    tools: dict[str, FastMcpTool]
    # Basic information
    profile: str
    agent: Agent
    prompts: dict[str, str]
    observe_formats: dict[str, str]
    # Sub-worflows
    sub_workflows: dict[str, Workflow]
    
    def __init__(
        self, 
        profile: str, 
        prompts: dict[str, str] = {}, 
        observe_formats: dict[str, str] = {}, 
        sub_workflows: dict[str, Workflow] = {}, 
        *args, 
        **kwargs,
    ) -> None:
        """Initialize the BaseWorkflow.

        Args:
            profile (str):
                The profile of the workflow.
            prompts (dict[str, str], optional):
                The prompts of the workflow. The key is the prompt name and the value is the prompt content. 
            observe_formats (dict[str, str], optional):
                The formats of the observation. The key is the observation name and the value is the format method name. 
            sub_workflows (dict[str, Workflow], optional):
                The sub-workflows of the workflow. The key is the name of the sub-workflow and the value is the 
                sub-workflow instance. 
            *args:
                The arguments to be passed to the parent class.
            **kwargs:
                The keyword arguments to be passed to the parent class.
        """
        # Initialize the parent class
        super().__init__(*args, **kwargs)
                
        # Initialize the workflow components
        self.profile = profile
        self.prompts = prompts
        self.observe_formats = observe_formats
        self.sub_workflows = sub_workflows
        
        # Initialize the agent
        self.agent = None

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
