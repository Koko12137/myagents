from enum import Enum
from typing import Protocol, runtime_checkable, List
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class DocumentLogAction(Enum):
    """DocumentLogAction is the action of the document log.
    
    Attributes:
        INSERT (str):
            The action of inserting a log.
        DELETE (str):
            The action of deleting a log.
        REPLACE (str):
            The action of replacing a log.
    """
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"


class FormatType(Enum):
    """FormatType defines the format options for the document.
    
    Attributes:
        LINE_NUMBER (str):
            Format as "line_number: content" for each line.
        ARTICLE (str):
            Format as complete article with all content.
        ORIGINAL (str):
            Format as original content (before any modifications).
    """
    LINE_NUMBER = "line_number"
    ARTICLE = "article"
    ORIGINAL = "original"


class DocumentLog(BaseModel):
    """DocumentLog is the log of the document.
    
    Attributes:
        action (DocumentLogAction):
            The action of the log.
        line (int):
            The line of the log.
        content (str):
            The content of the log.
    """
    action: DocumentLogAction = Field(description="The action of the log.")
    line: int = Field(description="The line of the log.")
    content: str = Field(description="The content of the log.")


@runtime_checkable
class Document(Protocol):
    """Document is the protocol for the document.
    
    Attributes:
        logs (list[DocumentLog]):
            The logs of the document.
        original_lines (list[str]):
            The original lines of the document.
        current_lines (list[str]):
            The current lines of the document after applying logs.
    """
    logs: List[DocumentLog]
    original_lines: List[str]
    current_lines: List[str]
    
    def modify(self, logs: List[DocumentLog]) -> None:
        """Apply a list of modification logs to the document.
        
        Args:
            logs: List of DocumentLog to apply
        """
        ...
    
    def reset_to_original(self) -> None:
        """Reset the document to its original state by clearing all logs."""
        ...
    
    def format(self, format_type: FormatType = FormatType.ARTICLE) -> str:
        """Format the document according to the specified format type.
        
        Args:
            format_type: The format type (LINE_NUMBER, ARTICLE, or ORIGINAL)
            
        Returns:
            The formatted document content
        """
        ...


@dataclass
class BaseDocument:
    """BaseDocument is a concrete implementation of the Document protocol.
    
    This class provides functionality to:
    - Store original document content split into lines
    - Apply modifications through logs without changing the original
    - Generate current content by applying logs to original
    - Provide a modify interface for accepting diff inputs
    - Format content in different ways through a single interface
    
    Attributes:
        original_content (str):
            The original document content as a string.
        original_lines (List[str]):
            The original document split into lines.
        logs (List[DocumentLog]):
            The list of modification logs.
        current_lines (List[str]):
            The current state of the document after applying logs.
    """
    original_content: str
    original_lines: List[str] = field(init=False)
    logs: List[DocumentLog] = field(default_factory=list)
    current_lines: List[str] = field(init=False)
    
    def __post_init__(self):
        """Initialize the document by splitting content into lines."""
        self.original_lines = self.original_content.splitlines(keepends=True)
        self.current_lines = self.original_lines.copy()
    
    def modify(self, logs: List[DocumentLog]) -> None:
        """Apply a list of modification logs to the document.
        
        Args:
            logs: List of DocumentLog to apply
        """
        for log in logs:
            # Validate line number
            if log.line < 1 or log.line > len(self.current_lines) + 1:
                raise ValueError(f"Line number {log.line} is out of range (1-{len(self.current_lines) + 1})")
            self.logs.append(log)
            self._apply_log(log)
    
    def _apply_log(self, log: DocumentLog) -> None:
        """Apply a single log entry to the current lines.
        
        Args:
            log: The log entry to apply
        """
        line_idx = log.line - 1  # Convert to 0-indexed
        
        if log.action == DocumentLogAction.INSERT:
            # Insert content at the specified line
            if line_idx <= len(self.current_lines):
                self.current_lines.insert(line_idx, log.content)
            else:
                # Append if line number is beyond current length
                self.current_lines.append(log.content)
                
        elif log.action == DocumentLogAction.DELETE:
            # Delete the specified line
            if line_idx < len(self.current_lines):
                self.current_lines.pop(line_idx)
                
        elif log.action == DocumentLogAction.REPLACE:
            # Replace the specified line
            if line_idx < len(self.current_lines):
                self.current_lines[line_idx] = log.content
            else:
                # If line doesn't exist, insert it
                while len(self.current_lines) < line_idx:
                    self.current_lines.append("")
                self.current_lines.append(log.content)
    
    def reset_to_original(self) -> None:
        """Reset the document to its original state by clearing all logs."""
        self.logs.clear()
        self.current_lines = self.original_lines.copy()
    
    def get_logs(self) -> List[DocumentLog]:
        """Get all modification logs.
        
        Returns:
            List of all modification logs
        """
        return self.logs.copy()
    
    def apply_diff(self, diff_logs: List[DocumentLog]) -> None:
        """Apply a list of diff logs to the document.
        
        Args:
            diff_logs: List of diff logs to apply
        """
        self.modify(diff_logs)
    
    def format(self, format_type: FormatType = FormatType.ARTICLE) -> str:
        """Format the document according to the specified format type.
        
        Args:
            format_type: The format type (LINE_NUMBER, ARTICLE, or ORIGINAL)
            
        Returns:
            The formatted document content
        """
        if format_type == FormatType.LINE_NUMBER:
            return self._format_as_line_number()
        elif format_type == FormatType.ARTICLE:
            return self._format_as_article()
        elif format_type == FormatType.ORIGINAL:
            return self._format_as_original()
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
    
    def _format_as_line_number(self) -> str:
        """Format the document as "line_number: content" for each line.
        
        Returns:
            The document formatted with line numbers
        """
        formatted_lines = []
        for i, line in enumerate(self.current_lines, 1):
            # Remove trailing newline for display, but keep the line number format
            content = line.rstrip('\n')
            formatted_lines.append(f"{i}: {content}")
        
        return "\n".join(formatted_lines)
    
    def _format_as_article(self) -> str:
        """Format the document as a complete article.
        
        Returns:
            The complete article content
        """
        return "".join(self.current_lines)
    
    def _format_as_original(self) -> str:
        """Format the document as original content.
        
        Returns:
            The original content of the document
        """
        return "".join(self.original_lines)
