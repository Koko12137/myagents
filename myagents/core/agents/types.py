from enum import Enum


class AgentType(Enum):
    """The type of the agent.
    
    Attributes:
        REACT (str):
            The reason and act agent. This agent works on a basic reason and act workflow. 
        ORCHESTRATE (str):
            The orchestrator agent. This agent works on an objective and key outputs orchestration workflow. 
        PLAN_AND_EXECUTE (str):
            The plan and executor agent. This agent works on a plan and executor workflow. 
    """
    REACT = "ReActAgent"
    ORCHESTRATE = "OrchestrateAgent"
    PLAN_AND_EXECUTE = "PlanAndExecuteAgent"
