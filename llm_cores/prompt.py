"""
Prompt storage classes for the REPL.
"""
from typing import Optional, List, Any


class Prompt:
    """
    Represents a stored input prompt with metadata.
    
    Attributes:
        text: The raw input text
        index: Sequential index in the prompt history
        result: Optional result from inspection
    """
    
    def __init__(self, text: str, index: int):
        self.text = text
        self.index = index
        self.result: Any = None
        self.tags: List[str] = []
    
    def __repr__(self) -> str:
        preview = self.text[:40] + "..." if len(self.text) > 40 else self.text
        preview = preview.replace('\n', '\\n')
        return f"Prompt[{self.index}]({preview!r})"
    
    def __str__(self) -> str:
        return self.text
    
    def tag(self, *tags: str) -> "Prompt":
        """Add tags to this prompt for later filtering."""
        self.tags.extend(tags)
        return self
    
    def has_tag(self, tag: str) -> bool:
        """Check if prompt has a specific tag."""
        return tag in self.tags


class PromptList:
    """
    A collection of prompts with filtering and lookup capabilities.
    """
    
    def __init__(self):
        self._prompts: List[Prompt] = []
    
    def add(self, text: str) -> Prompt:
        """Add a new prompt to the list."""
        prompt = Prompt(text, index=len(self._prompts))
        self._prompts.append(prompt)
        return prompt
    
    def __getitem__(self, key) -> Prompt:
        """Access prompts by index (supports negative indexing)."""
        return self._prompts[key]
    
    def __len__(self) -> int:
        return len(self._prompts)
    
    def __iter__(self):
        return iter(self._prompts)
    
    def __repr__(self) -> str:
        return f"PromptList({len(self._prompts)} prompts)"
    
    def filter(self, tag: Optional[str] = None) -> List[Prompt]:
        """Filter prompts by tag."""
        result = self._prompts
        if tag:
            result = [p for p in result if p.has_tag(tag)]
        return result
    
    @property
    def last(self) -> Optional[Prompt]:
        """Get the most recent prompt."""
        return self._prompts[-1] if self._prompts else None

