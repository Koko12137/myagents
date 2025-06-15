import asyncio

from myagents.tests.test import test_async_query
from scripts.sample.sampling import main as sampling_main


def main():
    # asyncio.run(test_async_query())
    asyncio.run(sampling_main())


if __name__ == "__main__":
    main()
    