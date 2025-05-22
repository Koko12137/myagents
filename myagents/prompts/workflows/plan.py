PLAN_REASON_PROMPT = """
## Stage Description
You are now in the planning stage. You should decide the action to be taken in this task, and You can orchestrate sub 
tasks according to the available tools. 

## Task Description
Now you should decide the action to be taken in this task, and You can orchestrate sub tasks according to the available 
tools, but you will not be provided with the accessability to the tools, so you don't need to provide the tool calls but 
only the task orchestration and your reason. 
    1. If you think one sub task can be finished directly without any sub-tasks, you should mark this sub task as 
        a leaf task.
    2. If you think this task cannot be finished directly and it needs to be split into sub-tasks, you should 
        mark this task as a non-leaf task and it will be orchestrated by the next planning. 
        
## Attention
You should not call any tools in the sub-tasks, but only the task orchestration and your reason. And you should only 
take care of the next one level of sub-tasks without considering the higher level of sub-tasks. In your sub tasks, you 
should provide the task dependencies and the task correct condition. You can decide your choice according the tools that 
are available.

## Tools 
All the available tools are listed below:
{tools}

## Format Constraints
You must follow the following format, otherwise you will be penalized. 
<think> 
your thinking if this part is needed
</think>
<reason>
1. reason 1
2. reason 2
3. ...
</reason>
<task_finish_condition>
All of the sub-tasks should be finished or any one of them is finished. 
</task_finish_condition>
<task>
- [ ] create task 1
- [ ] create task 2
    - Description:
        The description of the sub-task. 
    - Dependencies:
        task 1
    - Is Leaf:
        True or False (Whether the task is a leaf task)
    - Correct Condition:
        The correct condition of the sub-task.
- [ ] ...
</task>

## Task Context Observation
You can observe the task context below, it including the task question, description, status, strategy, parent task information 
and sub-tasks information that are already created.

{task_context}
"""


EXEC_PLAN_PROMPT = """
## Task Description
Now you will be provided with some tools that can modify the orchestration of sub-tasks, and you should decide the 
tool call and its arguments according to the history messages or you can reply that this task can be finished directly. 
The following is your choice:
1. You can reply that this task can be finished directly without calling any tools, and this task is a leaf task. 
2. You can reply the tool call and its arguments. 
3. All the task orchestration are finished. End the loop of planning.

## Attention
You SHOULD ONLY orchestrate one sub layer of sub-tasks for the current task. You do not need to consider the higher 
level of sub-tasks. All the rest of the sub-tasks will be orchestrated by the next planning. 

## Tools
All the available tools are listed below:
{tools}

## Format Constraints
You must follow the following format, otherwise you will be penalized. 
<think> 
your thinking if this part is needed
</think>
<reason>
1. reason 1
2. reason 2
3. ...
</reason>
<task>
- [x] task 1 created
- [ ] now working on task 2 creating 
    - Dependencies:
        task 1
    - Description:
        The description of the sub-task.  
    - Is Leaf:
        True or False (Whether the task is a leaf task)
    - Correct Condition:
        The correct condition of the sub-task. 
- [ ] ...
</task>
<description>
The description of the task. In this task, whether or not you need a utility, and what information you want to get if you do. 
</description>

## Task Context Observation
You can observe the task context below, it including the task question, description, status, strategy, parent task information 
and sub-tasks information that are already created.

{task_context}
"""