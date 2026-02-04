"""
Prompt storage classes for the witt library.
"""
from typing import Optional, List, Any, Tuple, Union, Dict, overload
import functools

from .state_node import StateNode
from .computational_node import ComputationalNode, ActivationRef, ConstantNode


@functools.lru_cache(maxsize=1)
def _get_byte_decoder() -> Dict[str, int]:
    """
    Build the reverse mapping from GPT-2 BPE unicode characters to bytes.
    This is the inverse of the byte_encoder used in GPT-2/BPE tokenizers.
    """
    # Characters that map to themselves (printable ASCII + some Latin-1)
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    
    # Other bytes map to characters starting at U+0100
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    
    # Return unicode char -> byte mapping
    return {chr(c): b for b, c in zip(bs, cs)}


def decode_bpe_token(token_str: str) -> str:
    """
    Decode a BPE token string to its actual text representation.
    Handles all GPT-2 style byte-level encodings (Ġ for space, Ċ for newline, 
    em-dashes, curly quotes, etc.)
    """
    byte_decoder = _get_byte_decoder()
    try:
        # Convert each character to its byte value, then decode as UTF-8
        byte_values = bytes([byte_decoder.get(c, ord(c)) for c in token_str])
        return byte_values.decode('utf-8', errors='replace')
    except Exception:
        # Fallback: return as-is if decoding fails
        return token_str


class Prompt:
    """
    Represents a stored input prompt with metadata.
    
    Attributes:
        tokens: The tokenized sequence as (token_id, token_str) tuples
        text: (property) The raw input text, reconstructed from tokens
        uid: Unique internal ID based on creation order (immutable)
        result: Optional result from inspection
    """
    
    # Class-level counter for unique IDs
    _next_uid: int = 0
    
    def __init__(self, tokens: Optional[List[Tuple[int, str]]] = None):
        self.tokens = tokens or []
        self.result: Any = None
        
        # Assign unique internal ID and increment counter
        self.uid = Prompt._next_uid
        Prompt._next_uid += 1
        
        # Initialize state with uid as prompt_index
        self.head: StateNode = StateNode(prompt_index=self.uid, parent=None)
    
    @property
    def text(self) -> str:
        """Reconstruct text from tokens by joining token strings."""
        return ''.join(decode_bpe_token(t[1]) for t in self.tokens)

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

    def __getitem__(self, token_idx: int):
        if token_idx < -len(self.tokens) or token_idx >= len(self.tokens):
            raise IndexError(f"Token index {token_idx} out of range [-{len(self.tokens)}, {len(self.tokens)})")
        return TokenProxy(self, token_idx % len(self.tokens))

    def append(self, new_tokens: List[Tuple[int, str]]):
        self.tokens.extend(new_tokens)

class TokenProxy:
    """Proxy for accessing token-level operations on a prompt."""
    
    def __init__(self, prompt: Prompt, index: int):
        self.prompt = prompt
        self.index = index
        
    def __repr__(self) -> str:
        decoded = decode_bpe_token(self.prompt.tokens[self.index][1])
        return f"Token({self.index}, {decoded!r})"

    def __getitem__(self, layer_idx: int):
        return LayerProxy(self.prompt, self.index, layer_idx)


class LayerProxy:
    """Proxy for accessing layer-level operations on a token."""
    
    def __init__(self, prompt: Prompt, token_idx: int, layer_idx: int):
        self.prompt = prompt
        self.token_idx = token_idx
        self.layer_idx = layer_idx 
        
    def __repr__(self) -> str:
        decoded = decode_bpe_token(self.prompt.tokens[self.token_idx][1])
        return f"Layer({self.layer_idx}, Token({self.token_idx}, {decoded!r}))"

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
    def add(self, tokens: List[Tuple[int, str]]) -> Prompt: ...
    
    def add(self, tokens_or_prompt: Union[List[Tuple[int, str]], Prompt]) -> Prompt:
        """
        Add a new prompt to the collection.
        
        Args:
            tokens_or_prompt: Either a list of (token_id, token_str) tuples or an existing Prompt object
            
        Returns:
            The added Prompt object
        """
        if isinstance(tokens_or_prompt, Prompt):
            prompt = tokens_or_prompt
        else:
            prompt = Prompt(tokens=tokens_or_prompt)
        
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
    
    @property
    def last(self) -> Optional[Prompt]:
        """Get the most recently added prompt."""
        if not self._prompts:
            return None
        # Dict maintains insertion order in Python 3.7+
        return list(self._prompts.values())[-1]

