from enum import Enum

from myagents.core.envs.base import BaseEnvironment
from myagents.core.envs.query import Query


class EnvironmentType(Enum):
    """EnvironmentType is the type of the environment.
    
    Attributes:
        QUERY (EnvironmentType):
            The type of the Query environment.
    """
    QUERY = "Query"


__all__ = ["BaseEnvironment", "Query", "EnvironmentType"]
