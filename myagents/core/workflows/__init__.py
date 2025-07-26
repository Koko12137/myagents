from enum import Enum

from .base import BaseWorkflow
from .react import BaseReActFlow, TreeTaskReActFlow
from .plan_and_exec import PlanAndExecFlow
from .orchestrate import OrchestrateFlow


class WorkflowType(Enum):
    """WorkflowType is the type of the workflow.
    
    Attributes:
        REACT (str):
            The type of the ReActFlow.
        PLAN_AND_EXEC (str):
            The type of the PlanAndExecFlow.
        ORCHESTRATE (str):
            The type of the OrchestrateFlow.
    """
    REACT = "ReActFlow"
    PLAN_AND_EXEC = "PlanAndExecFlow"
    ORCHESTRATE = "OrchestrateFlow"


__all__ = [
    "BaseWorkflow", 
    "BaseReActFlow", 
    "PlanAndExecFlow", 
    "OrchestrateFlow", 
    "TreeTaskReActFlow", 
    "WorkflowType",
]
