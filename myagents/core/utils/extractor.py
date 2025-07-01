import re


def extract_by_label(content: str, *labels: str) -> str:
    """Extract the content by the label.
    
    Args:
        content (str):
            The content to extract.
        *labels (str):
            The labels to extract.
            
    Returns:
        str:
            The extracted content. If the content is not found, return an empty string. 
    """
    # Traverse all the labels
    for label in labels:
        
        # Try with closing tag
        result = re.search(f"<{label}>\s*\n(.*)\n\s*</{label}>", content, re.DOTALL)
        if result:
            return result.group(1)
        else:
            # Try without closing tag
            result = re.search(f"<{label}>\s*\n(.*)\n\s*", content, re.DOTALL)
            if result:
                return result.group(1)
            
    # All the labels are not found
    return ""
