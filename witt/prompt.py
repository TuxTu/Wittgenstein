"""
Prompt storage classes for the witt library.
"""
from typing import Optional, List, Any, Tuple, Union, Dict, overload

from .state_node import StateNode
from .computational_node import ComputationalNode, ActivationRef, ConstantNode


class Prompt:
    """
    Represents a stored input prompt with metadata.
    
    Attributes:
        text: The raw input text
        uid: Unique internal ID based on creation order (immutable)
        tokens: The tokenized sequence
        result: Optional result from inspection
    """
    
    # Class-level counter for unique IDs
    _next_uid: int = 0
    
    def __init__(self, text: str, tokens: Optional[List[Tuple[int, str]]] = None):
        self.text = text
        self.tokens = tokens or []
        self.result: Any = None
        
        # Assign unique internal ID and increment counter
        self.uid = Prompt._next_uid
        Prompt._next_uid += 1
        
        # Initialize state with uid as prompt_index
        self.head: StateNode = StateNode(prompt_index=self.uid, parent=None)
    
    @property
    def current_state_id(self):
        return self.head.time_step

    @property
    def token_ids(self) -> List[int]:
        """Return just the token IDs from the tokens list."""
        return [t[0] for t in self.tokens]

    def __repr__(self) -> str:
        preview = self.text[:40] + "..." if len(self.text) > 40 else self.text
        new_line = '\n'
        tab = '\t'
        preview = preview.replace(new_line, '\\n').replace(tab, '\\t')
        return f"Prompt[{self.uid}]({preview!r})"
    
    def __str__(self) -> str:
        return self.text
    
    def get_state_at(self, time_step: int) -> "StateNode":
        """Traverses history backwards to find the state at a specific time step."""
        curr = self.head
        
        if time_step > curr.time_step or time_step < 0:
            raise ValueError(f"Requested invalid time step {time_step} (Current head is {curr.time_step})")

        while curr.time_step > time_step:
            curr = curr.parent

        return curr
    
    def has_tag(self, tag: str) -> bool:
        """Check if prompt has a specific tag."""
        return tag in self.tags

    def __getitem__(self, token_idx: int):
        if token_idx < -len(self.tokens) or token_idx >= len(self.tokens):
            raise IndexError(f"Token index {token_idx} out of range [-{len(self.tokens)}, {len(self.tokens)})")
        return TokenProxy(self, token_idx % len(self.tokens))


class TokenProxy:
    """Proxy for accessing token-level operations on a prompt."""
    
    def __init__(self, prompt: "Prompt", index: int):
        self.prompt = prompt
        self.index = index
        
    def __repr__(self) -> str:
        new_line = '\n'
        tab = '\t'
        return f"Token({self.index}, {self.prompt.tokens[self.index][1].replace('Ġ', ' ').replace('Ċ', new_line).replace('ĉ', tab)!r})"

    def __getitem__(self, layer_idx: int):
        return LayerProxy(self.prompt, self.index, layer_idx)


class LayerProxy:
    """Proxy for accessing layer-level operations on a token."""
    
    def __init__(self, prompt: "Prompt", token_idx: int, layer_idx: int):
        self.prompt = prompt
        self.token_idx = token_idx
        self.layer_idx = layer_idx 
        
    def __repr__(self) -> str:
        new_line = '\n'
        tab = '\t'
        return f"Token({self.token_idx}, {self.prompt.tokens[self.token_idx][1].replace('Ġ', ' ').replace('Ċ', new_line).replace('ĉ', tab)!r})"

    def __getitem__(self, module: str):
        return ActivationRef(self.prompt.uid, self.prompt.current_state_id, self.token_idx, self.layer_idx, module)

    def __setitem__(self, module: str, value_node):
        # LHS: Check input
        if not isinstance(value_node, ComputationalNode):
            # Allow implicit conversion of constants: p[0] = 5.0
            value_node = ConstantNode(value_node) 

        # Create NEW State
        new_state = StateNode(
            prompt_index=self.prompt.uid,
            parent=self.prompt.head,
            patch_target=(self.layer_idx, self.token_idx, module),
            patch_value_node=value_node  # Store the lazy math
        )
        
        # Advance the pointer
        self.prompt.head = new_state


class PromptList:
    """A collection of prompts with lookup by uid."""
    
    def __init__(self):
        self._prompts: Dict[int, Prompt] = {}  # Keyed by uid
    
    @overload
    def add(self, prompt: Prompt) -> Prompt: ...
    
    @overload
    def add(self, text: str, tokens: Optional[List[Tuple[int, str]]] = None) -> Prompt: ...
    
    def add(self, text_or_prompt: Union[str, Prompt], tokens: Optional[List[Tuple[int, str]]] = None) -> Prompt:
        """
        Add a new prompt to the collection.
        
        Args:
            text_or_prompt: Either a text string or an existing Prompt object
            tokens: Optional tokenized sequence (only used when text is provided)
            
        Returns:
            The added Prompt object
        """
        if isinstance(text_or_prompt, Prompt):
            prompt = text_or_prompt
        else:
            prompt = Prompt(text_or_prompt, tokens=tokens)
        
        self._prompts[prompt.uid] = prompt
        return prompt
    
    def __getitem__(self, uid: int) -> Prompt:
        """Access prompts by uid."""
        return self._prompts[uid]
    
    def __contains__(self, uid: int) -> bool:
        """Check if a prompt with the given uid exists."""
        return uid in self._prompts
    
    def __len__(self) -> int:
        return len(self._prompts)
    
    def __iter__(self):
        return iter(self._prompts.values())
    
    def __repr__(self) -> str:
        return f"PromptList({len(self._prompts)} prompts)"
    
    def filter(self, tag: Optional[str] = None) -> List[Prompt]:
        """Filter prompts by tag."""
        result = list(self._prompts.values())
        if tag:
            result = [p for p in result if p.has_tag(tag)]
        return result
    
    @property
    def last(self) -> Optional[Prompt]:
        """Get the most recently added prompt."""
        if not self._prompts:
            return None
        # Dict maintains insertion order in Python 3.7+
        return list(self._prompts.values())[-1]

