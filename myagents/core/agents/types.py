from enum import Enum


class AgentType(Enum):
    """代理的类型
    
    属性:
        PROXY (str):
            代理代理。这个代理是用户的代理。
        REACT (str):
            推理和行动代理。这个代理在基本的推理和行动工作流上工作。
        TREE_REACT (str):
            树形推理和行动代理。这个代理在树形推理和行动工作流上工作。
        ORCHESTRATE (str):
            编排代理。这个代理在编排工作流上工作。
        PLAN_AND_EXECUTE (str):
            计划和执行代理。这个代理在计划和执行工作流上工作。
    """
    PROXY = "ProxyAgent"
    REACT = "ReActAgent"
    TREE_REACT = "TreeReActAgent"
    ORCHESTRATE = "OrchestrateAgent"
    PLAN_AND_EXECUTE = "PlanAndExecuteAgent"
