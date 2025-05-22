import json
import os

from myagents.src.envs.query import Query
from myagents.src.agents.base import BaseStepCounter
from myagents.src.interface import OrchestratedFlows
from myagents.src.factory import AutoAgent, OrchestratedFlowsConfig
from myagents.src.utils.logger import init_logger


async def test_async_query():
    question = "请你讲一讲陀思妥耶夫斯基的罪与罚"
    description = "我需要知道陀思妥耶夫斯基的罪与罚结构化的内容"
    
    # Load API keys from json and convert to environment variables
    with open("configs/api_keys.json", "r") as f:
        api_keys = json.load(f)
        for key, value in api_keys.items():
            os.environ[key] = value
    
    # Create a list of agents according to the config file
    with open("configs/agents.json", "r") as f:
        json_data = json.load(f)
    agents = OrchestratedFlowsConfig.model_validate(json_data)
    # Build a global step counter
    step_counter = BaseStepCounter(100)
    # Create a logger
    logger = init_logger(
        sink="stdout", 
        level="DEBUG", 
    )
    # Create Factory
    factory = AutoAgent()
    # Build ReActFlow
    react_flow: OrchestratedFlows = factory.build_orchestrated_workflows(
        config=agents, 
        step_counter=step_counter, 
        custom_logger=logger, 
        debug=True,
    )

    # Create a new Query
    query = Query(
        agent=react_flow.agent,
        react_flow=react_flow,
        custom_logger=logger, 
        debug=True,
    )
    # Run the query
    answer = await query.run(question, description)
    # Print the answer
    print(answer)
