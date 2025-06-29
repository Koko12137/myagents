import asyncio

import click

from myagents.tests.test_query import test_async_query
from scripts.sample.sampling import main as sampling_main
from scripts.sample.neg_sampling import main as neg_sampling_main


@click.command()
@click.option("--func", type=click.Choice(["test", "sampling", "neg_sampling"]), help="Run function")
def main(func: str):
    if func == "test":
        asyncio.run(test_async_query())
    elif func == "sampling":
        asyncio.run(sampling_main())
    elif func == "neg_sampling":
        asyncio.run(neg_sampling_main())
    else:
        raise ValueError(f"Invalid function: {func}")


if __name__ == "__main__":
    main()
    