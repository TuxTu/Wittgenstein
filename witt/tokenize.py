"""
Tokenization utilities for the witt library.
"""
import functools
from typing import Dict, List, Tuple


@functools.lru_cache(maxsize=1)
def _get_byte_decoder() -> Dict[str, int]:
    """
    Build the reverse mapping from GPT-2 BPE unicode characters to bytes.
    This is the inverse of the byte_encoder used in GPT-2/BPE tokenizers.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]

    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    return {chr(c): b for b, c in zip(bs, cs)}


def decode_bpe_token(token_str: str) -> str:
    """
    Decode a BPE token string to its actual text representation.
    Handles all GPT-2 style byte-level encodings (Ġ for space, Ċ for newline,
    em-dashes, curly quotes, etc.)
    """
    byte_decoder = _get_byte_decoder()
    try:
        byte_values = bytes([byte_decoder.get(c, ord(c)) for c in token_str])
        return byte_values.decode('utf-8', errors='replace')
    except Exception:
        return token_str


def tokenize(tokenizer, prompt) -> List[Tuple[int, str]]:
    """
    Analyzes how a specific prompt is broken down into tokens.

    Args:
        tokenizer: The tokenizer to use
        prompt: The text to tokenize

    Returns:
        A list of (Token ID, String Representation) tuples.
    """
    # 1. Get the numerical IDs
    input_ids = tokenizer.encode(prompt)

    # 2. Get the string representation (visual tokens)
    # We use convert_ids_to_tokens to see special characters like 'Ġ' (space)
    token_strs = tokenizer.convert_ids_to_tokens(input_ids)

    # Zip them together for easy inspection
    result = list(zip(input_ids, token_strs))
    return result
