import asyncio

import click

from scripts.sample.sampling import main as sampling_main
from scripts.sample.neg_sampling import main as neg_sampling_main
from scripts.tests.raw_llm import main as raw_llm_main


@click.group()
def cli():
    """
    Main CLI group.
    """
    pass

@cli.command()
@click.option("--dataset_path", type=str, required=True, help="数据集路径")
@click.option("--output_dir", type=str, required=True, help="输出目录")
@click.option("--output_name", type=str, required=True, help="输出文件名，必须是jsonl文件名")
@click.option("--input_field", type=str, default="question", show_default=True, help="输入字段名")
@click.option("--key_config", type=str, default="configs/api_keys.json", show_default=True, help="密钥配置文件路径")
@click.option("--agents_config", type=str, default="configs/sampling.json", show_default=True, help="智能体配置文件路径")
@click.option("--num_workers", type=int, default=1, show_default=True, help="工作线程数")
def sampling(dataset_path: str, output_dir: str, output_name: str, input_field: str, key_config: str, agents_config: str, num_workers: int):
    """
    运行样本处理
    """
    assert output_name.endswith('.jsonl'), "输出文件必须是jsonl文件"
    asyncio.run(sampling_main(
        dataset_path=dataset_path,
        output_dir=output_dir,
        output_name=output_name,
        input_field=input_field,
        key_config=key_config,
        agents_config=agents_config,
        num_workers=num_workers,
    ))

@cli.command()
@click.option("--dataset-path", type=str, required=True, help="数据集路径")
@click.option("--output-file", type=str, required=True, help="输出文件路径")
@click.option("--input-field", type=str, default="question", show_default=True, help="输入字段名")
@click.option("--output-field", type=str, default="answer", show_default=True, help="输出字段名")
@click.option("--concurrency", type=int, default=1, show_default=True, help="并发数")
def test_raw_llm(dataset_path: str, output_file: str, input_field: str, output_field: str, concurrency: int):
    """
    测试raw_llm
    """
    asyncio.run(raw_llm_main())

if __name__ == "__main__":
    cli()
    