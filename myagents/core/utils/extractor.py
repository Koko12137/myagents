import re
from typing import Optional


def extract_by_label(content: str, *labels: str) -> Optional[str]:
    """Extract the content by the label.
    
    Args:
        content (str):
            The content to extract.
        labels (str):
            The labels to extract.
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
    return None
