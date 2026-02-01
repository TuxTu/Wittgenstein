from witt import load_model, load_tokenizer, tokenize, Executor, Prompt, PromptList
from ui import inspector

model_id = "Qwen/Qwen3-0.6B"

model = load_model(model_id, use_cpu=True)
tokenizer = load_tokenizer(model_id)

inspect = inspector.PromptInspector(model_id)
prompts = PromptList()

messages = [
    {"role": "user", "content": "I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns – “thoughts” – into your mind. I want to assess how well you can detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials."},
    {"role": "assistant", "content": "Ok."},
    {"role": "user", "content": "Ok, let's get started.\nTrial 1: Do you detect an injected thought? If so, what is the injected thought about"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)

print(text)

tokens = tokenize(tokenizer, text)
prompt = Prompt(tokens)
prompts.add(prompt)
inspect.inspect(prompt)

executor = Executor(model, tokenizer, prompts)

print(executor.generate(prompt, max_new_tokens=256))