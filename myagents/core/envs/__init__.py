from enum import Enum

from myagents.core.envs.base import BaseEnvironment
from myagents.core.envs.complex_query import ComplexQuery
from myagents.core.envs.orchestrate import Orchestrate


class EnvironmentType(Enum):
    """环境的类型
    
    属性:
        COMPLEX_QUERY (EnvironmentType):
            复杂询问环境的类型
        ORCHESTRATE (EnvironmentType):
            编排环境的类型
    """
    COMPLEX_QUERY = "ComplexQuery"
    ORCHESTRATE = "Orchestrate"


__all__ = ["BaseEnvironment", "ComplexQuery", "Orchestrate", "EnvironmentType"]
