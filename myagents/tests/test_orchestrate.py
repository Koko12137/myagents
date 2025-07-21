import json
import os

from datasets import load_from_disk

from myagents.core.envs.orchestrate import Orchestrate
from myagents.core.factory import AutoAgent, AutoAgentConfig


async def test_async_orchestrate():
    # question = "请你讲一讲陀思妥耶夫斯基的罪与罚"
    # description = "我需要知道陀思妥耶夫斯基的罪与罚结构化的内容"
    # 读取 datasets/GAOKAO-Math-Bench 数据集
    dataset = load_from_disk("datasets/GAOKAO-Math-Bench")
    # 随机选择一个数据集
    data = dataset.shuffle(seed=42).select(range(1))
    # 打印数据集
    print(data[0])
    
    # Load API keys from json and convert to environment variables
    with open("configs/api_keys.json", "r") as f:
        api_keys = json.load(f)
        for key, value in api_keys.items():
            os.environ[key] = value
    
    # Create a list of agents according to the config file
    with open("configs/sampling.json", "r") as f:
        config = AutoAgentConfig.model_validate(json.load(f))
    
    # Create Factory
    factory = AutoAgent()
    # Build ReActFlow
    orchestrate: Orchestrate = factory.auto_build(config=config)
    
    # Run the query
    blueprint, todo_view, json_view = await orchestrate.run(
        question="请仔细分析这道数学题，给出正确的答案选项。最终答案只包含A,B,C,D中的一个", 
        description=data[0]["question"],
        sub_task_depth=1,
    )
    print("-" * 60)
    # Check the history, blueprint and todo view
    print("blueprint: ", blueprint)
    print("todo view: ", todo_view)
    print("json view: ", json_view)
