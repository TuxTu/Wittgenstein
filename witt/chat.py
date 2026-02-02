from typing import Optional, List, Tuple, Union
from typing_extensions import TypedDict
from enum import Enum

from .prompt import Prompt, TokenProxy


class ChatMessage(TypedDict):
    role: str
    content: str

class MessageProxy:
    """Proxy for accessing token-level operations within a specific message."""
    
    def __init__(self, chat: "Chat", start_idx: int, end_idx: int):
        self.chat = chat
        self.start_idx = start_idx  # Global token index (inclusive)
        self.end_idx = end_idx      # Global token index (exclusive)
        self.length = end_idx - start_idx
    
    def __getitem__(self, relative_token_idx: int) -> TokenProxy:
        """Access a token by relative index within this message."""
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
        return f"MessageProxy(tokens[{self.start_idx}:{self.end_idx}], len={self.length})"

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
        
        start_idx, end_idx = self.messages[message_idx]
        return MessageProxy(self.chat, start_idx, end_idx)
    
    def __len__(self) -> int:
        return len(self.messages)
    
    def __repr__(self) -> str:
        return f"RoleProxy(role='{self.role}', messages={len(self.messages)})"


class Chat(Prompt):

    class RoleType(Enum):
        SYSTEM = "system"
        USER = "user"
        ASSISTANT = "assistant"
        THINKING = "thinking"

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

            start_idx, end_idx = self.extract_message(message["content"], remaining_tokens)
            if start_idx == -1 or end_idx == -1:
                raise ValueError(f"Content not found in given tokens: {message['content']!r}...")
            self.messages_in_tokens[role].append((idx_shift + start_idx, idx_shift + end_idx + 1))
            idx_shift += end_idx + 1
            remaining_tokens = remaining_tokens[end_idx+1:]
        
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

        if not tokens or not content:
            return (-1, -1)
        
        char_to_token = []
        reconstructed = ""
        
        for token_idx, token_str in enumerate(tokens):
            decoded = token_str.replace('Ġ', ' ').replace('Ċ', '\n').replace('ĉ', '\t')
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