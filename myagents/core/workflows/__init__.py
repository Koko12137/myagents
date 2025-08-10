from enum import Enum

from .base import BaseWorkflow
from .memory import MemoryCompressWorkflow, EpisodeMemoryFlow
from .react import BaseReActFlow, TreeTaskReActFlow, MemoryReActFlow, MemoryTreeTaskReActFlow
from .plan import PlanWorkflow, MemoryPlanWorkflow
from .plan_and_exec import PlanAndExecFlow, MemoryPlanAndExecFlow
from .orchestrate import OrchestrateFlow, MemoryOrchestrateFlow


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
    "WorkflowType",
    "BaseWorkflow", 
    "MemoryCompressWorkflow", "EpisodeMemoryFlow", 
    "BaseReActFlow", "TreeTaskReActFlow", "MemoryReActFlow", "MemoryTreeTaskReActFlow", 
    "PlanWorkflow", "MemoryPlanWorkflow", 
    "PlanAndExecFlow", "MemoryPlanAndExecFlow", 
    "OrchestrateFlow", "MemoryOrchestrateFlow", 
]
