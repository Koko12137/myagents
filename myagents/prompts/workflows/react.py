REASON_PROMPT = """
You now need to reason about the task and give a general orchestration and action plan for executing the task.
Your result in this stage will be used to guide the orchestrating the sub-tasks in later stages. The root task 
will be split into several sub-tasks and also the sub-tasks can be split into several sub-sub-tasks if needed. 

## What you should do
- Reason about the task and give a general orchestration and action plan for executing the task.
- Orchestrate the hierarchical structure of the task and the sub-tasks. 
- Ensure the message you need in each sub-task and whether the sub-task is a leaf task or not.

## Format Constraint
<think>
your thoughts here
</think>
<orchestration>
your hierarchical orchestration here
1. task 1
    - The information you need
    - The sub-task is a leaf task or not
    1. sub-task 1
        - The information you need
        - The sub-task is a leaf task or not
    2. sub-task 2
        - The information you need
        - The sub-task is a leaf task or not
        1. sub-sub-task 1
        2. sub-sub-task 2
2. task 2
    - The information you need
    - The sub-task is a leaf task or not
    1. sub-task 3
    2. sub-task 4
</orchestration>

## Task Context
The task context is as follows: 
{task_context}
"""


ACTION_FAILURE_PROMPT = """
Now there is a failure in the task. You need to reason about the failure and select the best action to fix failure 
or to skip and re-orchestrate the task.

## Constraint
- You can only retry one failed sub-task up to 3 times. If the limit is reached, the task will be cancelled. 
- You can not skip the task if there is no parent task. 

## Failure
{failure}
"""


SUMMARY_PROMPT = """
Now one sub-task is finished, you need to check and summarize the result of the task. 

## Sub-task
{sub_task}
"""