TASK_INFER_PROMPT = """
## Question
{question}

## Parent Plan
{parent_plan}

## Dependencies
{dependencies}

## Sub-Plans
{sub_plans}
""" 

TASK_RESULT_PROMPT = """
## Question
{question}

## Answer
{answer}
""" 


TOOL_CALL_PROMPT = """
## Status
{status}

## Result
{result}
""" 
