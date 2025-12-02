import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_id, use_fp16=True):
    """
    Loads a Hugging Face model with memory optimizations for local inference.
    
    Args:
        model_id (str): The specific model name (e.g., "Qwen/Qwen3-0.6B-Instruct")
        device_map (str): "auto" to split across GPU/CPU, or specific device "cuda:0".
        use_fp16 (bool): Whether to use float16 (half precision) to save RAM.
    
    Returns:
        model, tokenizer
    """
    print(f"\r[-] Loading model: {model_id}...")
    
    # Select precision
    dtype = torch.float16 if use_fp16 else torch.float32

    if torch.cuda.is_available():
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
            trust_remote_code=True, # Needed for Qwen/custom architectures
            low_cpu_mem_usage=True  # Speeds up loading
        )

        if device == "mps":
            model.to(device)

        print(f"\r[+] Model loaded successfully on {model.device}")
        return model
    
    except Exception as e:
        print(f"\r[!] Error loading model: {e}")
        raise e

def load_tokenizer(model_id):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print(f"\r[+] Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        print(f"\r[!] Error loading tokenizer: {e}")
        raise e
