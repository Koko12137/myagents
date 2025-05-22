import asyncio

from myagents.tests.test import test_async_query


def main():
    asyncio.run(test_async_query())


if __name__ == "__main__":
    main()
    