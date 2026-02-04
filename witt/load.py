import torch
import re
from typing import List, Union, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class TokenizerWrapper:
    """
    A wrapper around HuggingFace tokenizers that provides custom apply_chat_template
    to properly handle thinking content in assistant messages.
    
    Thinking content should be in the 'reasoning_content' field of assistant messages,
    and will be wrapped with <think>...</think> tags.
    
    All other attributes and methods are passed through to the underlying tokenizer.
    """
    
    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"
    
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._supports_thinking = self._detect_thinking_support()
        self._start_token, self._end_token = self._detect_chat_tokens()
    
    def _detect_thinking_support(self) -> bool:
        """Check if the tokenizer's chat template supports thinking."""
        template = getattr(self._tokenizer, 'chat_template', None)
        if not template:
            return False
        return "<think>" in template or "reasoning_content" in template or "enable_thinking" in template
    
    def _detect_chat_tokens(self) -> tuple:
        """Detect the start/end tokens from the chat template (called once at init)."""
        template = getattr(self._tokenizer, 'chat_template', '')
        
        if '<|im_start|>' in template:
            # Qwen/ChatML style
            return '<|im_start|>', '<|im_end|>'
        elif '[INST]' in template:
            # Llama style
            return '[INST]', '[/INST]'
        else:
            # Fallback to ChatML
            return '<|im_start|>', '<|im_end|>'
    
    @property
    def supports_thinking(self) -> bool:
        """Whether this model supports thinking traces."""
        return self._supports_thinking
    
    def __getattr__(self, name):
        """Pass through all other attributes to the underlying tokenizer."""
        return getattr(self._tokenizer, name)
    
    def __call__(self, *args, **kwargs):
        """Pass through tokenizer calls."""
        return self._tokenizer(*args, **kwargs)
    
    def _has_reasoning_content(self, messages: List[dict]) -> bool:
        """Check if any assistant message has reasoning_content."""
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("reasoning_content"):
                return True
        return False
    
    def apply_chat_template(
        self,
        messages: List[dict],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs
    ) -> Union[str, List[int]]:
        """
        Apply chat template with proper handling of thinking content.
        
        If an assistant message has 'reasoning_content' field, it will be
        wrapped with <think>...</think> tags and included in the output.
        
        Args:
            messages: List of message dicts with 'role', 'content', and optional 'reasoning_content'
            tokenize: Whether to return token IDs or string
            add_generation_prompt: Whether to add generation prompt
            **kwargs: Additional arguments passed to underlying apply_chat_template
            
        Returns:
            Formatted string or token IDs
        """
        # If no reasoning_content in any message, just use the original tokenizer
        if not self._has_reasoning_content(messages):
            return self._tokenizer.apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, **kwargs
            )
        
        # Has reasoning_content - build template ourselves to ensure it's preserved
        result = self._build_chat_template(messages, add_generation_prompt)
        
        if tokenize:
            return self._tokenizer.encode(result, add_special_tokens=False)
        return result
    
    def _build_chat_template(self, messages: List[dict], add_generation_prompt: bool) -> str:
        """
        Build chat template with reasoning_content properly formatted.
        """
        parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            reasoning = msg.get("reasoning_content", "")
            
            if role == "system":
                parts.append(f"{self._start_token}system\n{content}{self._end_token}\n")
            elif role == "user":
                parts.append(f"{self._start_token}user\n{content}{self._end_token}\n")
            elif role == "assistant":
                if reasoning:
                    # Include thinking content
                        full_content = f"{self.THINK_OPEN}\n{reasoning}{self.THINK_CLOSE}\n{content}"
                else:
                    full_content = content
                parts.append(f"{self._start_token}assistant\n{full_content}{self._end_token}\n")
        
        if add_generation_prompt:
            parts.append(f"{self._start_token}assistant\n")
        
        return "".join(parts)


def load_model(model_id, use_fp16=True, use_cpu=False):
    """
    Loads a Hugging Face model with memory optimizations for local inference.
    
    Args:
        model_id (str): The specific model name (e.g., "Qwen/Qwen3-0.6B-Instruct")
        use_fp16 (bool): Whether to use float16 (half precision) to save RAM.
    
    Returns:
        model: The loaded model
    """
    print(f"\r[-] Loading model: {model_id}...")
    
    # Select precision
    dtype = torch.float16 if use_fp16 else torch.float32

    if use_cpu:
        device = "cpu"
        print("[!] User requested to use CPU for inference.")
    elif torch.cuda.is_available():
        device = "cuda"
        print("[-] NVIDIA GPU detected.")
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    #     print("[-] Apple Silicon (MPS) detected. Using GPU acceleration.")
    else:
        device = "cpu"
        print("[!] No GPU detected. Falling back to CPU.")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device,
            trust_remote_code=True,  # Needed for Qwen/custom architectures
            low_cpu_mem_usage=True   # Speeds up loading
        )

        if device == "mps":
            model.to(device)

        print(f"\r[+] Model loaded successfully on {model.device}")
        return model
    
    except Exception as e:
        print(f"\r[!] Error loading model: {e}")
        raise e


def load_tokenizer(model_id) -> TokenizerWrapper:
    """
    Load a tokenizer for the specified model, wrapped with thinking support.
    
    Returns a TokenizerWrapper that:
    - Has all the same methods/attributes as the original tokenizer
    - Provides a custom apply_chat_template that preserves thinking content
    - Auto-detects whether the model supports thinking
    
    Args:
        model_id: The HuggingFace model ID
        
    Returns:
        TokenizerWrapper instance
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        wrapped = TokenizerWrapper(tokenizer)
        
        thinking_status = "with thinking support" if wrapped.supports_thinking else "no thinking support"
        print(f"\r[+] Tokenizer loaded successfully ({thinking_status})")
        
        return wrapped
    except Exception as e:
        print(f"\r[!] Error loading tokenizer: {e}")
        raise e