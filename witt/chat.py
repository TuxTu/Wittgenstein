"""
Chat storage classes for the witt library.
"""
from typing import Optional, List, Tuple, Dict, Union

from .prompt import Prompt, TokenProxy


class ChatRoleProxy:
    """
    Proxy for accessing tokens belonging to a specific role in a chat.
    
    Enables indexing pattern: chat["user"][token_idx][layer_idx][module]
    where token_idx is relative to this role's tokens.
    """
    
    def __init__(self, chat: "Chat", role: str, token_ranges: List[Tuple[int, int]]):
        self.chat = chat
        self.role = role
        self.token_ranges = token_ranges
        # Build flat list of absolute token indices for this role
        self._token_indices: List[int] = []
        for start, end in token_ranges:
            self._token_indices.extend(range(start, end))
    
    def __getitem__(self, idx: int) -> TokenProxy:
        """
        Access token by index within this role's tokens.
        
        Args:
            idx: Token index relative to this role (supports negative indexing)
            
        Returns:
            TokenProxy for the specified token
        """
        if idx < 0:
            idx = len(self._token_indices) + idx
        if idx < 0 or idx >= len(self._token_indices):
            raise IndexError(f"Token index {idx} out of range for role '{self.role}' (has {len(self._token_indices)} tokens)")
        
        actual_token_idx = self._token_indices[idx]
        return TokenProxy(self.chat, actual_token_idx)
    
    def __len__(self) -> int:
        """Number of tokens belonging to this role."""
        return len(self._token_indices)
    
    def __iter__(self):
        """Iterate over TokenProxy objects for this role."""
        for i in range(len(self)):
            yield self[i]
    
    def __repr__(self) -> str:
        return f"ChatRoleProxy(role='{self.role}', tokens={len(self)})"


class Chat(Prompt):
    """
    Represents a chat conversation as a single prompt.
    
    Internally stores messages as a list of {"role": ..., "content": ...} dicts,
    compatible with tokenizer.apply_chat_template(). After template application,
    the entire chat becomes a single tokenized sequence.
    
    Access patterns:
        chat["system"][token_idx][layer_idx][module] - system tokens
        chat["user"][token_idx][layer_idx][module] - user tokens
        chat["assistant"][token_idx][layer_idx][module] - assistant tokens
        chat["thinking"][token_idx][layer_idx][module] - thinking tokens (if enabled)
        chat[token_idx][layer_idx][module] - direct token access (absolute index)
    
    Thinking mode:
        Some models (e.g., Qwen3) support a "thinking" mode that adds chain-of-thought
        reasoning. This is DISABLED by default. Set enable_thinking=True to enable it.
    
    Example:
        chat = Chat()
        chat.add_system("You are a helpful assistant.")
        chat.add_user("Hello!")
        chat.add_assistant("Hi there!")
        
        # Tokenize (thinking disabled by default)
        chat.tokenize(tokenizer)
        
        # Or with thinking enabled
        chat.tokenize(tokenizer, enable_thinking=True)
        
        # Access tokens by role
        chat["user"][-1][10]["resid_post"] = some_value
    """
    
    # Valid roles for token access
    VALID_ROLES = ("system", "user", "assistant", "thinking")
    
    def __init__(self, id: int = 0, enable_thinking: bool = False):
        # Initialize parent with empty text/tokens - will be set by apply_template
        super().__init__(text="", id=id, tokens=None)
        
        # Whether thinking mode is enabled (affects tokenization)
        self.enable_thinking = enable_thinking
        
        # Message storage: [{"role": "system", "content": "..."}, ...]
        self._messages: List[Dict[str, str]] = []
        
        # Token range mapping: role -> list of (start, end) token index ranges
        # Allows multiple messages per role (e.g., multi-turn conversation)
        self._role_token_ranges: Dict[str, List[Tuple[int, int]]] = {
            "system": [],
            "user": [],
            "assistant": [],
            "thinking": []  # For chain-of-thought reasoning tokens
        }
    
    @property
    def messages(self) -> List[Dict[str, str]]:
        """
        Get messages in the format expected by tokenizer.apply_chat_template().
        
        Returns:
            List of {"role": str, "content": str} dicts
        """
        return self._messages
    
    def add_system(self, content: str) -> "Chat":
        """
        Add or replace the system message.
        
        Args:
            content: The system message content
            
        Returns:
            self for method chaining
        """
        # Remove existing system message if present
        self._messages = [m for m in self._messages if m["role"] != "system"]
        # Insert system at the beginning
        self._messages.insert(0, {"role": "system", "content": content})
        return self
    
    def add_user(self, content: str) -> "Chat":
        """
        Add a user message to the conversation.
        
        Args:
            content: The user message content
            
        Returns:
            self for method chaining
        """
        self._messages.append({"role": "user", "content": content})
        return self
    
    def add_assistant(self, content: str) -> "Chat":
        """
        Add an assistant message to the conversation.
        
        Args:
            content: The assistant message content
            
        Returns:
            self for method chaining
        """
        self._messages.append({"role": "assistant", "content": content})
        return self
    
    def apply_template(
        self, 
        tokens: List[Tuple[int, str]], 
        role_token_ranges: Dict[str, List[Tuple[int, int]]],
        rendered_text: Optional[str] = None,
        enable_thinking: Optional[bool] = None
    ) -> "Chat":
        """
        Apply tokenization results after using tokenizer.apply_chat_template().
        
        This should be called after tokenizing the chat to set up the token
        sequence and role-to-token mappings.
        
        Args:
            tokens: The full tokenized sequence as list of (token_id, token_str)
            role_token_ranges: Mapping from role to list of (start, end) token ranges
                              e.g., {"user": [(5, 15), (25, 35)], "thinking": [(35, 50)]}
            rendered_text: Optional rendered template text
            enable_thinking: Whether thinking mode was used during tokenization
            
        Returns:
            self for method chaining
        """
        self.tokens = tokens
        self._role_token_ranges = role_token_ranges
        if rendered_text is not None:
            self.text = rendered_text
        if enable_thinking is not None:
            self.enable_thinking = enable_thinking
        return self
    
    def __getitem__(self, key: Union[str, int]):
        """
        Access by role name or token index.
        
        Args:
            key: Either a role string ("system", "user", "assistant", "thinking") 
                 or an integer token index
                 
        Returns:
            For string: ChatRoleProxy for the specified role
            For int: TokenProxy for the specified token (absolute index)
        """
        if isinstance(key, str):
            if key not in self.VALID_ROLES:
                raise KeyError(f"Unknown role: {key}. Valid roles: {self.VALID_ROLES}")
            return ChatRoleProxy(self, key, self._role_token_ranges.get(key, []))
        elif isinstance(key, int):
            # Direct token access (inherited behavior)
            return super().__getitem__(key)
        else:
            raise TypeError(f"Invalid key type: {type(key)}. Expected str (role) or int (token index)")
    
    @property
    def system(self) -> ChatRoleProxy:
        """Direct access to system tokens."""
        return ChatRoleProxy(self, "system", self._role_token_ranges.get("system", []))
    
    @property
    def user(self) -> ChatRoleProxy:
        """Direct access to user tokens."""
        return ChatRoleProxy(self, "user", self._role_token_ranges.get("user", []))
    
    @property
    def assistant(self) -> ChatRoleProxy:
        """Direct access to assistant tokens."""
        return ChatRoleProxy(self, "assistant", self._role_token_ranges.get("assistant", []))
    
    @property
    def thinking(self) -> ChatRoleProxy:
        """Direct access to thinking/reasoning tokens (only populated when enable_thinking=True)."""
        return ChatRoleProxy(self, "thinking", self._role_token_ranges.get("thinking", []))
    
    def __repr__(self) -> str:
        role_counts = {}
        for msg in self._messages:
            role = msg["role"]
            role_counts[role] = role_counts.get(role, 0) + 1
        
        parts = []
        for role in ["system", "user", "assistant"]:
            if role in role_counts:
                parts.append(f"{role_counts[role]} {role}")
        
        msg_info = ", ".join(parts) if parts else "empty"
        token_info = f", {len(self.tokens)} tokens" if self.tokens else ""
        thinking_info = ", thinking=on" if self.enable_thinking else ""
        return f"Chat({msg_info}{token_info}{thinking_info})"
    
    def __len__(self) -> int:
        """Total number of tokens in the chat (after template application)."""
        return len(self.tokens) if self.tokens else 0
