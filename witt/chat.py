from typing import Optional, List, Tuple, Union, Dict
from typing_extensions import TypedDict, NotRequired
from enum import Enum

from .prompt import Prompt, TokenProxy, decode_bpe_token


class ChatMessage(TypedDict):
    role: str
    content: str
    reasoning_content: NotRequired[str]


class ContentProxy:
    """Proxy for accessing token-level operations within a content or reasoning_content span."""
    
    def __init__(self, chat: "Chat", start_idx: int, end_idx: int, field: str):
        self.chat = chat
        self.start_idx = start_idx  # Global token index (inclusive)
        self.end_idx = end_idx      # Global token index (exclusive)
        self.length = end_idx - start_idx
        self.field = field  # "content" or "reasoning_content"
    
    def __getitem__(self, relative_token_idx: int) -> TokenProxy:
        """Access a token by relative index within this content span."""
        # Handle negative indices
        if relative_token_idx < 0:
            relative_token_idx = self.length + relative_token_idx
        
        if relative_token_idx < 0 or relative_token_idx >= self.length:
            raise IndexError(f"Token index {relative_token_idx} out of range [0, {self.length})")
        
        # Translate to global token index
        global_token_idx = self.start_idx + relative_token_idx
        return self.chat.prompt_getitem(global_token_idx)
    
    def __len__(self) -> int:
        return self.length
    
    def __repr__(self) -> str:
        return f"ContentProxy({self.field}, tokens[{self.start_idx}:{self.end_idx}], len={self.length})"


class MessageProxy:
    """
    Proxy for accessing a message with optional reasoning_content.
    
    Access patterns:
    - msg["content"] → ContentProxy for content tokens
    - msg["reasoning_content"] → ContentProxy for reasoning tokens (if exists)
    - msg[token_idx] → TokenProxy (shortcut for msg["content"][token_idx])
    """
    
    def __init__(self, chat: "Chat", token_ranges: Dict[str, Tuple[int, int]]):
        self.chat = chat
        self.token_ranges = token_ranges  # {"content": (start, end), "reasoning_content": (start, end) or None}
        
        # Calculate total length (content only for backward compatibility)
        content_range = token_ranges.get("content")
        self.length = content_range[1] - content_range[0] if content_range else 0
    
    @property
    def has_reasoning(self) -> bool:
        """Check if this message has reasoning_content."""
        return "reasoning_content" in self.token_ranges and self.token_ranges["reasoning_content"] is not None
    
    def __getitem__(self, key: Union[str, int]) -> Union[ContentProxy, TokenProxy]:
        """
        Access by field name or token index.
        
        - msg["content"] → ContentProxy for content
        - msg["reasoning_content"] → ContentProxy for reasoning (raises if not present)
        - msg[5] → TokenProxy for token at index 5 in content
        """
        if isinstance(key, str):
            if key == "content":
                start, end = self.token_ranges["content"]
                return ContentProxy(self.chat, start, end, "content")
            elif key == "reasoning_content":
                if not self.has_reasoning:
                    raise KeyError("This message has no reasoning_content")
                start, end = self.token_ranges["reasoning_content"]
                return ContentProxy(self.chat, start, end, "reasoning_content")
            else:
                raise KeyError(f"Unknown field: {key}. Use 'content' or 'reasoning_content'")
        elif isinstance(key, int):
            # Shortcut: index directly into content
            return self["content"][key]
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
    
    def __len__(self) -> int:
        """Length of content tokens (for backward compatibility)."""
        return self.length
    
    def __repr__(self) -> str:
        content_range = self.token_ranges.get("content", (0, 0))
        reasoning_str = ""
        if self.has_reasoning:
            r_range = self.token_ranges["reasoning_content"]
            reasoning_str = f", reasoning[{r_range[0]}:{r_range[1]}]"
        return f"MessageProxy(content[{content_range[0]}:{content_range[1]}]{reasoning_str})"

class RoleProxy:
    """Proxy for accessing messages of a specific role."""
    
    def __init__(self, chat: "Chat", role: str):
        self.chat = chat
        self.role = role
        self.messages = chat.messages_in_tokens.get(role, [])
    
    def __getitem__(self, message_idx: int) -> MessageProxy:
        """Access a message by index within this role."""
        if message_idx < -len(self.messages) or message_idx >= len(self.messages):
            raise IndexError(f"Message index {message_idx} out of range for role '{self.role}' (has {len(self.messages)} messages)")
        
        token_ranges = self.messages[message_idx]
        return MessageProxy(self.chat, token_ranges)
    
    def __len__(self) -> int:
        return len(self.messages)
    
    def __repr__(self) -> str:
        return f"RoleProxy(role='{self.role}', messages={len(self.messages)})"


class Chat(Prompt):

    class RoleType(Enum):
        SYSTEM = "system"
        USER = "user"
        ASSISTANT = "assistant"

    def __init__(self, messages: Optional[List[ChatMessage]] = None, tokens: Optional[List[Tuple[int, str]]] = None):
        super().__init__()  # Initialize with empty tokens

        self.messages_in_tokens = {}
        
        if messages and tokens:
            self.append(messages, tokens)

    def append(self, new_messages: List[ChatMessage], new_tokens: List[Tuple[int, str]]):
        if not new_messages or not new_tokens:
            return
            
        remaining_tokens = [t[1] for t in new_tokens] if new_tokens else []
        idx_shift = len(self.tokens)

        for message in new_messages:
            role = message["role"]
            assert role in [r.value for r in self.RoleType], f"Unknown role: {role}"
            
            if role not in self.messages_in_tokens:
                self.messages_in_tokens[role] = []

            token_ranges: Dict[str, Tuple[int, int]] = {}
            
            # Extract reasoning_content if present
            reasoning_content = message.get("reasoning_content")
            if reasoning_content:
                r_start, r_end = self.extract_message(reasoning_content, remaining_tokens)
                if r_start == -1 or r_end == -1:
                    raise ValueError(f"reasoning_content not found in tokens: {reasoning_content[:50]!r}...")
                token_ranges["reasoning_content"] = (idx_shift + r_start, idx_shift + r_end + 1)
            
            # Extract content
            content = message["content"]
            c_start, c_end = self.extract_message(content, remaining_tokens)
            if c_start == -1 or c_end == -1:
                raise ValueError(f"Content not found in given tokens: {content[:50]!r}...")
            token_ranges["content"] = (idx_shift + c_start, idx_shift + c_end + 1)
            
            self.messages_in_tokens[role].append(token_ranges)
            
            # Advance past the last token we found (content comes after reasoning)
            idx_shift += c_end + 1
            remaining_tokens = remaining_tokens[c_end + 1:]
        
        super().append(new_tokens)

    def __getitem__(self, key: Union[str, int]) -> Union[RoleProxy, TokenProxy]:
        """
        Access by role name or token index.
        
        - chat["user"] → RoleProxy for user messages
        - chat[5] → TokenProxy for token at index 5 (same as Prompt behavior)
        """
        if isinstance(key, str):
            # Access by role name
            if key not in [r.value for r in self.RoleType]:
                raise KeyError(f"Unknown role: {key}")
            return RoleProxy(self, key)
        elif isinstance(key, int):
            # Fall back to Prompt's token indexing
            return self.prompt_getitem(key)
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def prompt_getitem(self, token_idx: int) -> TokenProxy:
        """Call parent Prompt's __getitem__ for token access."""
        return super().__getitem__(token_idx)

    def extract_message(self, content: str, tokens: List[str]) -> Tuple[int, int]:
        """
        Find the start and end token indices that contain the given message content.
        
        Uses full BPE byte decoding to properly handle all special characters
        (em-dashes, curly quotes, etc.)
        """
        if not tokens or not content:
            return (-1, -1)
        
        char_to_token = []
        reconstructed = ""
        
        for token_idx, token_str in enumerate(tokens):
            decoded = decode_bpe_token(token_str)
            for _ in decoded:
                char_to_token.append(token_idx)
            reconstructed += decoded
        
        pos = reconstructed.find(content)
        if pos == -1:
            return (-1, -1)
        
        # Map character positions to token indices
        start_idx = char_to_token[pos]
        end_idx = char_to_token[pos + len(content) - 1]
        
        return (start_idx, end_idx)