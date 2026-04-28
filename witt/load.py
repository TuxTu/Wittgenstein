"""
Model and tokenizer loading for the witt library.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .tokenizer_wrapper import TokenizerWrapper


def _select_device(use_cpu: bool = False) -> str:
    """Select the best available single-device runtime target."""
    if use_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _select_dtype(device: str, use_fp16: bool) -> torch.dtype:
    """Choose a safe dtype for the selected device."""
    if device == "cpu":
        return torch.float32
    return torch.float16 if use_fp16 else torch.float32


def _describe_model_device(model) -> str:
    """Best-effort string description of where the model lives."""
    try:
        return str(next(model.parameters()).device)
    except (AttributeError, StopIteration, TypeError):
        pass
    return str(getattr(model, "device", "unknown"))


def load_model(model_id, use_fp16=True, use_cpu=False):
    """
    Loads a Hugging Face model with memory optimizations for local inference.

    Args:
        model_id (str): The specific model name (e.g., "Qwen/Qwen3-0.6B-Instruct")
        use_fp16 (bool): Whether to use float16 (half precision) to save RAM.
        use_cpu (bool): Force CPU inference.

    Returns:
        model: The loaded model
    """
    print(f"\r[-] Loading model: {model_id}...")

    device = _select_device(use_cpu=use_cpu)
    dtype = _select_dtype(device, use_fp16)

    if device == "cpu" and use_cpu:
        print("[!] User requested to use CPU for inference.")
    elif device == "cuda":
        print("[-] NVIDIA GPU detected.")
    elif device == "mps":
        print("[-] Apple Silicon GPU detected.")
    else:
        print("[!] No GPU detected. Falling back to CPU.")

    try:
        load_kwargs = {
            "dtype": dtype,
            "trust_remote_code": True,   # Needed for Qwen/custom architectures
        }
        if device != "cpu":
            load_kwargs["low_cpu_mem_usage"] = True

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **load_kwargs,
        )

        if device in {"cuda", "mps"}:
            model.to(device)
        model.eval()

        print(f"\r[+] Model loaded successfully on {_describe_model_device(model)}")
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
