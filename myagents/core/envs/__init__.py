from enum import Enum

from myagents.core.envs.base import BaseEnvironment
from myagents.core.envs.complex_query import ComplexQuery
from myagents.core.envs.orchestrate import Orchestrate


class EnvironmentType(Enum):
    """EnvironmentType is the type of the environment.
    
    Attributes:
        QUERY (EnvironmentType):
            The type of the Query environment.
        COMPLEX_QUERY (EnvironmentType):
            The type of the ComplexQuery environment.
        ORCHESTRATE (EnvironmentType):
            The type of the Orchestrate environment.
    """
    COMPLEX_QUERY = "ComplexQuery"
    ORCHESTRATE = "Orchestrate"


__all__ = ["BaseEnvironment", "ComplexQuery", "Orchestrate", "EnvironmentType"]
