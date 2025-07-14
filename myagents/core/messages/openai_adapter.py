from typing import Union
import json

from myagents.core.messages import ToolCallRequest, ToolCallResult, AssistantMessage, UserMessage, SystemMessage


def to_openai_dict(
    messages: list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]
) -> list[dict[str, Union[str, dict]]]:
    """Convert the message to the OpenAI compatible messages dictionaries.
    
    Args:
        messages (list[Union[AssistantMessage, UserMessage, SystemMessage, ToolCallRequest, ToolCallResult]]): 
            The messages to convert.
            
    Returns:
        list[dict[str, Union[str, dict]]]: 
            The OpenAI compatible messages dictionaries.
    """
    # Create the generation history
    history = []
    for message in messages: 
        message_dict = {
            "role": message.role.value,
            "content": message.content
        }
        
        # This is only for OpenAI. 
        if isinstance(message, ToolCallResult):
            if message_dict['role'] == "tool":
                message_dict['tool_call_id'] = message.tool_call_id
                
        elif isinstance(message, SystemMessage) or isinstance(message, UserMessage):
            pass
                
        elif isinstance(message, AssistantMessage):
            # If the message is a tool call, add the tool call to the history
            if message.tool_calls != [] and message.tool_calls is not None:
                message_dict["tool_calls"] = []
                
                for tool_call in message.tool_calls:
                    message_dict["tool_calls"].append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name, 
                            "arguments": json.dumps(tool_call.args, ensure_ascii=False), 
                        }
                    })
        
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
            
        history.append(message_dict)
        
    return history
