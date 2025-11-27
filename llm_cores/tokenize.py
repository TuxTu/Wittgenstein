import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def tokenize(tokenizer, prompt):
    """
    Analyzes how a specific prompt is broken down into tokens.
    Returns a list of (Token ID, String Representation) tuples.
    """
    # 1. Get the numerical IDs
    input_ids = tokenizer.encode(prompt)
    
    # 2. Get the string representation (visual tokens)
    # We use convert_ids_to_tokens to see special characters like 'Ġ' (space)
    token_strs = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Zip them together for easy inspection
    result = list(zip(input_ids, token_strs))
    return result