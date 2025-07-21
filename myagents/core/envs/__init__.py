from enum import Enum

from myagents.core.envs.base import BaseEnvironment
from myagents.core.envs.query import Query
from myagents.core.envs.orchestrate import Orchestrate


class EnvironmentType(Enum):
    """EnvironmentType is the type of the environment.
    
    Attributes:
        QUERY (EnvironmentType):
            The type of the Query environment.
        ORCHESTRATE (EnvironmentType):
            The type of the Orchestrate environment.
    """
    QUERY = "Query"
    ORCHESTRATE = "Orchestrate"


__all__ = ["BaseEnvironment", "Query", "Orchestrate", "EnvironmentType"]
