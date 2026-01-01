"""
Chat completion functionality for the witt library.
"""
from typing import Union, List, Dict, Optional

import torch

from .chat import Chat


def complete_chat(
    model,
    tokenizer,
    chat: Union[Chat, List[Dict[str, str]]],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    enable_thinking: bool = False,
    return_full_response: bool = False,
) -> str:
    """
    Complete a chat conversation using the provided model.
    
    Args:
        model: The Hugging Face model to use for generation
        tokenizer: The tokenizer corresponding to the model
        chat: Either a Chat object or a list of message dicts 
              [{"role": "user", "content": "..."}, ...]
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling probability threshold
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling (False = greedy decoding)
        enable_thinking: Whether to enable thinking mode (for models like Qwen3)
        return_full_response: If True, return the full response including input.
                             If False (default), return only the generated part.
    
    Returns:
        The generated completion text
        
    Example:
        >>> model = load_model("Qwen/Qwen3-0.6B")
        >>> tokenizer = load_tokenizer("Qwen/Qwen3-0.6B")
        >>> 
        >>> chat = Chat()
        >>> chat.add_system("You are a helpful assistant.")
        >>> chat.add_user("What is the capital of France?")
        >>> 
        >>> response = complete_chat(model, tokenizer, chat)
        >>> print(response)
        "The capital of France is Paris."
    """
    # Convert Chat to messages list if needed
    if isinstance(chat, Chat):
        messages = chat.messages
    else:
        messages = chat
    
    # Apply chat template to format the conversation
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]
    
    # Move to model's device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate completion
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # Decode the output
    if return_full_response:
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        # Only decode the newly generated tokens
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text

