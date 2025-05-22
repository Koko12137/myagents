REFLECT_PROMPT = """
Now the task is finished or failed, you should reflect on the task context and summarize the task result. 

## Format Constraints 
You must follow the following format, or you will be punished:
<reflection> 
- Reflection 1: Error ... I can retry it later.
- Reflection 2: Error ... I should call for modify the orchestration.
- Reflection 3: This task is finished. 
- ...
</reflection>

<answer>
your answer
</answer>
"""