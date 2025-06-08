import asyncio

import click

from myagents.src.envs.query import Query
from myagents.src.agents.base import BaseStepCounter
from myagents.src.factory import load_agents_config, create_agent


@click.command()
@click.option("--question", type=str, help="The question to react.")
def main(question: str):
    asyncio.run(main_async(question))


async def main_async(question: str):
    # Create a list of agents according to the config file
    agents = load_agents_config("configs/agents.json")
    # Create a new global step counter
    step_counter = BaseStepCounter()
    # Create a new Query
    query = Query(
        react_agent=create_agent(agents.react_agent, step_counter),
        plan_agent=create_agent(agents.plan_agent, step_counter),
        action_agent=create_agent(agents.action_agent, step_counter),
    )
    # Run the query
    answer = await query.run(question)
    # Print the answer
    print(answer)


if __name__ == "__main__":
    main()
