import readline
from .input_processor import InputProcessor

def run():
    # 1. Configuration
    default_model = "Qwen/Qwen3-0.6B"
    model_id = input(f"\rEnter Model ID (default: {default_model}): ").strip()
    if not model_id:
        model_id = default_model

    # 2. Instantiate and Run
    processor = InputProcessor(model_id)
    processor.run()
