# Wittgenstein (Beta)

Wittgenstein (witt) is a token-centric library and REPL tool for inspecting and manipulating Large Language Model (LLM) activations.

## Overview

Wittgenstein provides an interactive environment for:
- **Tokenization inspection** — See exactly how your prompts are broken down into tokens
- **Activation patching** — Modify internal model activations during inference
- **Lazy computation graphs** — Build arbitrarily complex activation operations that are validated at definition time and evaluated on demand
- **Single-step inference** — Run one forward pass or generate one token at a time for fine-grained control
- **Chat mode** — Multi-turn conversation with the model, with full activation access to the chat history
- **Prompt management** — Store and organize multiple prompts for experimentation

## Installation

```bash
# Clone the repository
git clone https://github.com/TuxTu/Wittgenstein
cd Wittgenstein

# Install dependencies
pip install torch transformers accelerate
```

## Quick Start

```bash
python main.py
```

You'll be prompted to enter a model ID (defaults to `Qwen/Qwen3-0.6B`):

```
Enter Model ID (default: Qwen/Qwen3-0.6B):
```

Press Enter to use the default, or specify any Hugging Face model ID.

---

## Modes

Wittgenstein operates in three modes. Press **ESC** to cycle between them:
**COMMAND** → **INSTRUCT** → **CHAT** → **COMMAND** → ...

### INSTRUCT Mode (`>`)

Enter prompts to store them for later inspection and manipulation.

```
> The capital of Australia is
The capital of Australia is
·  ·       ·  ·         ·
```

When you enter a prompt, it is:
1. Tokenized using the model's tokenizer
2. Stored in the `prompts` list
3. Automatically inspected (tokenization displayed)

### COMMAND Mode (`>>>`)

Execute Python code in a persistent namespace with access to your prompts and the model.

```python
>>> prompts[0]
Prompt[0]('The capital of Australia is')

>>> len(prompts)
1
```

### CHAT Mode (`chat>`)

Multi-turn conversation with the loaded model. Each message is sent to the model, the response is displayed, and the full `Chat` object is stored in `prompts` for later manipulation in COMMAND mode.

```
chat> What is the capital of France?
The capital of France is Paris.

chat> What about Germany?
The capital of Germany is Berlin.
```

Switching away from CHAT mode resets the conversation. The `Chat` objects remain in `prompts` for inspection.

---

## Built-in Variables

| Variable | Description |
|----------|-------------|
| `prompts` | `PromptList` containing all stored prompts |
| `model` | The loaded Hugging Face model |
| `tokenizer` | The model's tokenizer |
| `env` | The execution environment |
| `inspector` | The `PromptInspector` instance |

## Built-in Functions

| Function | Description |
|----------|-------------|
| `inspect(p)` | Inspect a prompt's tokenization |
| `generate(p)` | Generate text continuation for a prompt |
| `last()` | Get the most recent prompt |
| `help` | Display help information |
| `struct` | Display model architecture |

---

## Working with Prompts

### Accessing Prompts

```python
>>> prompts[0]      # First prompt
>>> prompts[-1]     # Last prompt
>>> prompts.last    # Most recent prompt
>>> len(prompts)    # Number of stored prompts
```

### Prompt Properties

```python
>>> p = prompts[0]
>>> p.text          # The raw text
>>> p.tokens        # List of (token_id, token_string) tuples
>>> p.token_ids     # List of token IDs only
>>> p.uid           # Unique ID
```

### Inspecting Prompts

```python
>>> inspect(prompts[0])
```

This displays the tokenization with visual markers showing token boundaries.

---

## Activation Patching

Wittgenstein's core feature is the ability to patch activations during model inference.

### Activation addressing

Use bracket notation to reference specific activations:

```python
p[token_idx][layer_idx]["module"]
```

- `token_idx` — Token position (int, slice, or list; supports negative indexing)
- `layer_idx` — Layer number (int, slice, or list)
- `module` — One of: `"resid_pre"`, `"resid_post"`, `"mlp"`, `"attn"`

The indexing chain is split into two concepts:
- `ActivationView` while you are still selecting token/layer dimensions
- `ActivationAddress` / `ActivationAddressGroup` once you name a module

Call `.snapshot()` on an address to freeze the current causal write state into
an `ActivationRef` / `ActivationRefGroup` that behaves like a lazy tensor node.

```python
>>> p[0]               # ActivationView (token selected)
>>> p[0][5]            # ActivationView (token + layer selected)
>>> p[0][5]["resid_post"]  # ActivationAddress (stable coordinate)
>>> p[0][5]["resid_post"].snapshot()  # ActivationRef (frozen read)
```

### Identity-preserving activation addresses

Every `Prompt` maintains an address registry. Reading the same coordinate always
returns the **identical** `ActivationAddress` object:

```python
>>> addr1 = p[0][5]["resid_post"]
>>> addr2 = p[0][5]["resid_post"]
>>> addr1 is addr2
True
```

Overlapping range accesses share the same underlying atomic addresses:

```python
>>> group_a = p[0:5][3]["resid_post"]   # tokens 0,1,2,3,4
>>> group_b = p[3:7][3]["resid_post"]   # tokens 3,4,5,6
>>> group_a._addresses[3] is group_b._addresses[0]  # token 3 is shared
True
```

### Example: Reading an activation reference

```python
>>> p = prompts[0]
>>> addr = p[0][5]["resid_post"]
>>> addr
Addr(P0.T0.L5.resid_post)

>>> ref = addr.snapshot()
>>> ref
Ref(P0.T0.L5.resid_post[deps=0])
```

The `deps=N` suffix shows how many causal writes were captured in the
dependency snapshot at instantiation time (see below).

### Example: Patching an activation

```python
>>> p = prompts[0]
>>> q = prompts[1]

# Patch p's token-3 / layer-5 / resid_post with the activation from q
>>> p[3][5]["resid_post"] = q[2][5]["resid_post"].snapshot()

# Generate with the patch applied
>>> generate(p)
```

### Arithmetic on activations

All operations build a lazy computation graph validated at definition time via meta tensors:

```python
>>> q_ref = q[0][5]["resid_post"].snapshot()
>>> r_ref = r[0][5]["resid_post"].snapshot()
>>> p[0][5]["resid_post"] = q_ref * 2.0
>>> p[0][5]["resid_post"] = q_ref + r_ref
>>> p[0][5]["resid_post"] = (q_ref - r_ref) * 0.5
```

You can also compose operations into standalone expressions and evaluate them:

```python
>>> diff = q[0][5]["resid_post"].snapshot() - r[0][5]["resid_post"].snapshot()
>>> diff.eval()       # fills all leaf dependencies, returns concrete tensor
tensor([0.0312, -0.0421, ...])
```

Supported operations: `+`, `-`, `*`, `/`, `**`, `@` and all other PyTorch
tensor operations (via `__torch_function__`).

### Modification ledger and dependency snapshots

**Writes** (`p[tok][layer][module] = value`) are recorded in `p`'s
modification ledger.  They become active patches during the next forward
pass.  The latest write to a given position supersedes earlier ones.

**Reads** (`p[tok][layer][module]`) return a stable `ActivationAddress`.
Calling `.snapshot()` on that address creates an `ActivationRef` whose
**dependency snapshot** is frozen at snapshot time. The snapshot contains
exactly the writes in the source prompt's ledger that causally affect the
target position according to the *triangle rule*:

> A write at `(write_token, write_layer)` causally affects a target at
> `(target_token, target_layer)` iff
> `target_token >= write_token` AND `target_layer >= write_layer`.

This means that if you modify a prompt *after* extracting a reference from
it, the already-created reference retains its original snapshot and is
unaffected by the subsequent write:

```python
>>> q[1][3]["resid_post"] = original_value
>>> addr = q[2][5]["resid_post"]
>>> ref = addr.snapshot()   # snapshot captures original_value
>>> q[1][3]["resid_post"] = new_value   # too late -- ref already frozen
>>> p[3][5]["resid_post"] = ref
>>> generate(p)   # uses original_value for q's activation, not new_value
```

---

## Executor and Context Manager

The `Executor` is the engine that runs model inference with activation patching. It doubles as a context manager that injects real model metadata (hidden dimension, layer count) into the computational node system:

```python
from witt import Executor, Prompt, PromptList

executor = Executor(model, tokenizer, prompts)

with executor:
    # Inside the context, Prompt gains convenience methods:
    p.generate()          # generate text
    p.forward()           # single forward pass
    p.step()              # generate one token (greedy)

    # Evaluate any computational node:
    ref = p[0][5]["resid_post"].snapshot()
    val = ref.eval()      # fills dependencies, returns tensor

    # Composed expressions work too:
    diff = (
        p[0][5]["resid_post"].snapshot()
        - q[0][5]["resid_post"].snapshot()
    )
    val = diff.eval()

    # Or via the executor directly:
    val = executor.eval(some_node)
```

Inside the REPL, the context is entered automatically when you start the session.

### Single-step inference

For fine-grained control (e.g., agentic execution with per-token interventions):

```python
>>> # Single forward pass -- applies patches without generating
>>> p.forward()

>>> # Generate one token at a time
>>> token_id, token_str = p.step()
>>> print(token_str)
'Paris'
>>> token_id, token_str = p.step()  # prompt now has the new token appended
>>> print(token_str)
','
```

---

## Generation

Generate text continuations with optional activation patches:

```python
>>> p = prompts[0]
>>> generate(p)
'The capital of Australia is Sydney, which is known for...'
```

If you've applied patches to the prompt, they will be applied during generation.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **ESC** | Cycle modes: COMMAND → INSTRUCT → CHAT → COMMAND |
| **Enter** | Execute command / Store prompt / Send chat message |
| **Up/Down** | Navigate command history (per mode) |
| **Ctrl+C** | Exit the REPL |
| **q** / **quit** | Exit (in COMMAND mode) |

---

## Example Session

```
Enter Model ID (default: Qwen/Qwen3-0.6B):
[-] Loading model: Qwen/Qwen3-0.6B...
[+] Model loaded successfully on cpu

Starting in COMMAND mode. Press ESC to cycle modes (COMMAND → INSTRUCT → CHAT).
Type 'help' in COMMAND mode for available commands.

>>> # Press ESC to switch to INSTRUCT mode

> The capital of Australia is
The capital of Australia is
·  ·       ·  ·         ·

> The capital of Austria is
The capital of Austria is
·  ·       ·  ·         ·

>>> # Press ESC twice to get back to COMMAND mode

>>> len(prompts)
2

>>> p0 = prompts[0]
>>> p1 = prompts[1]

>>> # Before patch, despite Qwen3-0.6B gave the wrong capital of Australia
>>> generate(p0)
'The capital of Australia is Sydney...'

>>> generate(p1)
'The capital of Austria is Vienna...'

>>> # Patch p1's country token with p0's country token activation
>>> p1[3][0]["resid_post"] = p0[3][0]["resid_post"].snapshot()

>>> generate(p1)
[Dependency] Extracting 1 value(s) from P0 (snapshot writes: 0)
[Execute] Running P1 with 1 patch(es)...
'The capital of Austria is the city of Darwin...'  # Patched!
```

---

## Module Reference

### witt (Core Library)

| Module | Contents |
|--------|----------|
| `witt.prompt` | `Prompt`, `PromptList`, `ActivationView` |
| `witt.computational_node` | `ComputationalNode`, `ActivationAddress`, `ActivationAddressGroup`, `ActivationRef`, `ActivationRefGroup`, `ConstantNode`, `WriteRecord` |
| `witt.executor` | `Executor`, `get_active_executor()` |
| `witt.selector` | `Selector`, `IndexSelector`, `SliceSelector`, `ListSelector` |
| `witt.chat` | `Chat`, `ChatMessage`, `RoleProxy`, `MessageProxy`, `ContentProxy` |
| `witt.complete` | `complete_chat()` |
| `witt.load` | `load_model()`, `load_tokenizer()` |
| `witt.tokenizer_wrapper` | `TokenizerWrapper` |
| `witt.tokenize` | `tokenize()`, `decode_bpe_token()` |

### env (Execution Environment)

| Module | Contents |
|--------|----------|
| `env.environment` | `ExecutionEnvironment` — REPL runtime with context manager |

### ui (User Interface)

| Module | Contents |
|--------|----------|
| `ui.cli` | `run()` — Entry point |
| `ui.input_processor` | `InputProcessor` — REPL mode dispatch and orchestration |
| `ui.line_editor` | `LineEditor` — Single-line editing with history and paste |
| `ui.terminal` | Low-level terminal I/O, ANSI escape sequences |
| `ui.inspector` | `PromptInspector`, `InspectResult`, `LiveInspectDisplay` |
| `ui.screen_buffer` | `ScreenBuffer` — Terminal display management |

---

## License

See [LICENSE](LICENSE) for details.
