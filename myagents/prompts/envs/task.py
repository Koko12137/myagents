TASK_CREATED_PROMPT = """
- Task
- [ ] {question}
    - Description: 
        {description}
        
- Status
{status}
""" 

TASK_PLAN_PROMPT = """
- Task
- [ ] {question}
    - Description: 
        {description}
        
- Status
{status}
""" 

TASK_RUNNING_PROMPT = """
- Task
- [ ] {question}
    - Description: 
        {description}
        
- Status
{status}

- Parent Task Information with Status
{parent_task_information_with_status}

- Sub-Tasks Information with Status
{sub_tasks_information_with_status}
""" 

TASK_FINISHED_PROMPT = """
- Question
{question}

- Description
{description}

- Result
{result}
""" 

TASK_FAILED_PROMPT = """
- Question
{question}

- Description
{description}

- Status
{status}

- Parent Task Information with Strategy
{parent_task_information_with_strategy}

- Error Message
{error_message}
""" 

TASK_CANCELLED_PROMPT = """
- Question
{question}

- Description
{description}

- Status
{status}

- Parent Task Information with Strategy
{parent_task_information_with_strategy}

- Error Message
{error_message}
""" 
