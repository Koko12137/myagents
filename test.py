import asyncio

from scripts.tests.test_benchmark import main as benchmark_main
from scripts.tests.test_query import test_async_query
from scripts.tests.test_orchestrate import test_async_orchestrate
from scripts.sample.sampling import main as sampling_main
from scripts.sample.neg_sampling import main as neg_sampling_main


def main():
    # asyncio.run(test_async_query())
    asyncio.run(test_async_orchestrate())
    # asyncio.run(benchmark_main())
    # asyncio.run(sampling_main())
    # asyncio.run(neg_sampling_main())


if __name__ == "__main__":
    main()
    