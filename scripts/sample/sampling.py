import asyncio
import json
import os
import random
from typing import Union, List
from traceback import format_exc

from loguru import logger
from datasets import load_from_disk

from myagents.core.envs.orchestrate import Orchestrate
from myagents.core.factory import AutoAgent, AutoAgentConfig


PROMPT = """
# ORCHESTRATE —— EXEC —— 任务创建阶段

## 任务描述
你当前处于 “Orchestrate” 工作流中，你已经完成了任务的总体结构规划，现在需要根据规划的蓝图，创建任务。
【注意】：严令禁止直接回答任务，你只能思考和规划一步步应该怎么做。
【注意】：严令禁止修改总体规划，你只能根据总体规划，创建任务。
【注意】：最多有一层任务目标，即每个关键目标不能有子任务。这不意味着你只能创建一个任务目标，你可以
    创建多个任务目标（它们必须处于同一层）。

## 格式要求
这个阶段你不能思考，你只能按照蓝图规划的任务目标和关键产出，给出创建任务用的JSON。

## 任务总体规划蓝图

{blueprint}
"""


async def process_single_task(
    config: AutoAgentConfig, 
    description: str,
) -> list[dict[str, Union[str, dict]]]:
    """处理单个任务

    Args:
        config (AutoAgentConfig): 配置信息
    
    Returns:
        list[dict[str, Union[str, dict]]]: 
            处理结果，包含所有步骤的输出
    """
    # 创建工厂实例
    factory = AutoAgent()
    # 构建 Orchestrate Environment
    orchestrate: Orchestrate = factory.auto_build(config)
    
    try:
        # 运行 Orchestrate Environment
        blueprint, todo_view, json_view = await orchestrate.run(
            question="请仔细分析这道数学题，给出正确的答案选项。最终答案只包含A,B,C,D中的一个", 
            description=description,
            sub_task_depth=1,
        )
    except Exception as e:
        logger.error(e)
        pass
    
    return {
        "question": description,
        "prompt": PROMPT.format(blueprint=blueprint),
        "todo_view": todo_view,
        "json_view": json_view,
    }


async def run_sample(
    config: AutoAgentConfig,
    dataset_path: str,
    output_dir: str, 
    num_workers: int = 1, 
    max_samples: int = 300, 
    output_name: str = "results.jsonl", 
    input_field: str = "question", 
) -> List[dict[str, Union[str, dict]]]:
    """运行样本处理

    Args:
        config (AutoAgentConfig): 配置信息
        dataset_path (str): 数据集路径
        output_dir (str): 输出目录
        num_workers (int, optional): 工作线程数. 默认为1
        max_samples (int, optional): 最大样本数. 默认为500
        output_name (str, optional): 输出文件名. 默认为"results.jsonl"
        input_field (str, optional): 输入字段名. 默认为"question"

    Returns:
        List[dict[str, Union[str, dict]]]: 所有任务的处理结果
    """
    # 加载数据集
    dataset = load_from_disk(dataset_path)
    logger.info(f"加载数据集，数据集大小: {len(dataset)}")
    
    # 加载已经处理过的数据
    if os.path.exists(os.path.join(output_dir, output_name)):
        with open(os.path.join(output_dir, output_name), "r", encoding="utf-8") as f:
            processed_data = [json.loads(line) for line in f]
            
        # 过滤掉已经处理过的数据
        dataset = dataset.filter(lambda x: x[input_field] not in [item[input_field] for item in processed_data])
        logger.info(f"过滤掉已处理过的数据，剩余数据: {len(dataset)}")
    
    # 随机采样
    if len(dataset) > max_samples and max_samples > 0:
        indices = random.sample(range(len(dataset)), max_samples)
        dataset = dataset.select(indices)
    
    # 创建信号量来控制并发
    semaphore = asyncio.Semaphore(num_workers)
    # 创建jsonl文件
    jsonl_path = os.path.join(output_dir, output_name)
    
    async def process_with_semaphore(item):
        async with semaphore:
            try:
                result = await process_single_task(config, item[input_field])
                if result is not None:
                    # 将结果写入jsonl文件
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False)
                        f.write("\n")
                    
                    logger.info(f"已处理并保存: {item[input_field][:50]}...")
                return result
            except Exception as e:
                logger.error(f'处理提示词时发生错误: {item[input_field]}')
                logger.error(f'错误信息: {format_exc()}')
                return None
    
    # 创建所有任务
    tasks = [process_with_semaphore(item) for item in dataset]
    
    # 等待所有任务完成
    results = await asyncio.gather(*tasks)
    
    # 过滤掉None结果（错误的任务）
    return [r for r in results if r is not None]


async def main(
    dataset_path: str, 
    output_dir: str, 
    output_name: str, 
    input_field: str = "question", 
    key_config: str = "configs/api_keys.json", 
    agents_config: str = "configs/sampling.json", 
    num_workers: int = 1, 
) -> None:
    """运行样本处理
    
    Args:
        dataset_path (str): 
            数据集路径
        output_dir (str): 
            输出目录
        output_name (str): 
            输出文件名，必须是jsonl文件名
        input_field (str, optional): 
            输入字段名. 默认为"question"
        key_config (str, optional): 
            密钥配置文件路径. 默认为"configs/api_keys.json"
        agents_config (str, optional): 
            代理配置文件路径. 默认为"configs/sampling.json"
        num_workers (int, optional): 
            工作线程数. 默认为1
            
    Raises:
        AssertionError: 
            输出文件必须是jsonl文件
    """
    
    # 加载API密钥
    with open(key_config, "r") as f:
        api_keys = json.load(f)
        for key, value in api_keys.items():
            os.environ[key] = value
    
    # 加载配置
    with open(agents_config, "r") as f:
        config = AutoAgentConfig.model_validate(json.load(f))
    
    assert output_name.endswith(".jsonl"), "输出文件名必须是jsonl文件名"
    # 创建输出目录
    output_dir = f"{output_dir}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 运行样本处理
    results = await run_sample(
        config, 
        dataset_path, 
        output_dir, 
        num_workers, 
        max_samples=-1,
        output_name=output_name, 
        input_field=input_field, 
    )
    
    # 打印处理结果统计
    print(f"\n处理完成!")
    print(f"成功处理样本数: {len(results)}")
    print(f"结果已保存到jsonl文件中: {os.path.join(output_dir, output_name)}")


if __name__ == "__main__":
    asyncio.run(main())
    