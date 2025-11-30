"""
Prompt storage classes for the REPL.
"""
from typing import Optional, List, Any, Tuple
from .state_node import StateNode
from .computational_node import ComputationalNode, ActivationRef

class Prompt:
    """
    Represents a stored input prompt with metadata.
    
    Attributes:
        text: The raw input text
        id: Sequential index in the prompt history
        tokens: The tokenized sequence
        result: Optional result from inspection
    """
    
    def __init__(self, text: str, id: int, tokens: Optional[List[Tuple[int, str]]] = None):
        self.text = text
        self.id = id
        self.tokens = tokens or []
        self.result: Any = None
        self.tags: List[str] = []

        self.head: StateNode = StateNode(self.id, None)
    
    @property
    def current_state_id(self):
        return self.head.time_step

    @property
    def token_ids(self) -> List[int]:
        """Return just the token IDs from the tokens list."""
        return [t[0] for t in self.tokens]

    def __repr__(self) -> str:
        preview = self.text[:40] + "..." if len(self.text) > 40 else self.text
        preview = preview.replace('\n', '\\n')
        return f"Prompt[{self.id}]({preview!r})"
    
    def __str__(self) -> str:
        return self.text
    
    def tag(self, *tags: str) -> "Prompt":
        """Add tags to this prompt for later filtering."""
        self.tags.extend(tags)
        return self
    
    def has_tag(self, tag: str) -> bool:
        """Check if prompt has a specific tag."""
        return tag in self.tags

    def __getitem__(self, token_idx: int):
        if token_idx < -len(self.tokens) or token_idx >= len(self.tokens):
            raise IndexError(f"Token index {token_idx} out of range [-{len(self.tokens)}, {len(self.tokens)})")
        return TokenProxy(self, token_idx % len(self.tokens))

class TokenProxy:
    def __init__(self, prompt: "Prompt", index: int):
        self.prompt = prompt
        self.index = index
        
    def __repr__(self) -> str:
        return f"Token({self.index}, {self.prompt.tokens[self.index][1].replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')!r})"

    def __getitem__(self, layer_idx: int):
        return LayerProxy(self.prompt, self.index, layer_idx)

class LayerProxy:
    def __init__(self, prompt: "Prompt", token_idx: int, layer_idx: int):
        self.prompt = prompt
        self.token_idx = token_idx
        self.layer_idx = layer_idx 
        
    def __repr__(self) -> str:
        return f"Token({self.token_idx}, {self.prompt.tokens[self.token_idx][1].replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')!r})"

    def __getitem__(self, module: str):
        return ActivationRef(self.prompt.id, self.prompt.current_state_id, self.token_idx, self.layer_idx, module)

    def __setitem__(self, module: str, value_node):
        # LHS: Check input
        if not isinstance(value_node, ComputationalNode):
            # Allow implicit conversion of constants: p[0] = 5.0
            value_node = ConstantNode(value_node) 

        # Create NEW State
        new_state = StateNode(
            prompt_index=self.prompt.id,
            parent=self.prompt.head,
            patch_target=(self.layer_idx, self.token_idx, module),
            patch_value_node=value_node # Store the lazy math
        )
        
        # Advance the pointer
        self.prompt.head = new_state

class PromptList:
    """
    A collection of prompts with filtering and lookup capabilities.
    """
    
    def __init__(self):
        self._prompts: List[Prompt] = []
    
    def add(self, text: str, tokens: Optional[List[Tuple[int, str]]] = None) -> Prompt:
        """Add a new prompt to the list."""
        prompt = Prompt(text, id=len(self._prompts), tokens=tokens)
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

