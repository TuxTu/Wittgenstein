import sys
from llm_cores import *
import readline

class PromptInspector:
    def __init__(self, model_id):
        try:
            print(f"\n[-] Initializing {model_id}...")
            # Loading the model ensures we get the correct tokenizer config
            self.model, self.tokenizer = load_model(model_id)
        except Exception as e:
            print(f"\n[!] Critical Error: {e}")
            sys.exit(1)

    def inspect(self, prompt):
        """
        Takes a prompt, tokenizes it, and prints the visual breakdown.
        """
        # 1. Run the analysis logic (from loader.py)
        token_seq = tokenize(self.tokenizer, prompt)
        
        # 2. Display Result
        print(f"\n{'ID':<10} | {'Token String'}")
        print("-" * 30)
        
        for i, (token_id, token_str) in enumerate(token_seq):
            # We use repr() so you can see hidden characters like \n or spaces
            display_str = repr(token_str) 
            print(f"{i:<10} | {display_str}")

    def run(self):
        # 3. Main Interaction Loop
        while True:
            try:
                prompt = input("\n>>> ").strip()
                
                if prompt.lower() == 'q':
                    print("Exiting...")
                    break
                
                if not prompt:
                    continue

                # Delegate processing to the modular function
                self.inspect(prompt)
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"[!] Error processing prompt: {e}")

def main():
    # 1. Configuration
    default_model = "Qwen/Qwen3-0.6B"
    model_id = input(f"Enter Model ID (default: {default_model}): ").strip()
    if not model_id:
        model_id = default_model

    # 2. Instantiate and Run
    inspector = PromptInspector(model_id)
    inspector.run()

if __name__ == "__main__":
    main()