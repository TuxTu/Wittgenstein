# Wittgenstein(Beta)

Wittgenstein(witt) is a token-centric library and REPL tool for inspecting and manipulating Large Language Model (LLM) activations.

## Overview

Wittgenstein provides an interactive environment for:
- **Tokenization inspection** — See exactly how your prompts are broken down into tokens
- **Activation patching** — Modify internal model activations during inference
- **Prompt management** — Store and organize multiple prompts for experimentation

## Installation

```bash
# Clone the repository
git clone https://github.com/TuxTu/Wittgenstein
cd Wittgenstein

# Install dependencies
pip install torch transformers
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

Wittgenstein operates in two modes. Press **ESC** to switch between them.

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
>>> p.id            # Sequential index
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
p[token_idx][layer_idx][module]
```

- `token_idx` — Token position (int, slice, or list; supports negative indexing)
- `layer_idx` — Layer number (int, slice, or list)
- `module` — One of: `"resid_pre"`, `"resid_post"`, `"mlp"`, `"attn"`

### Identity-preserving activation references

Every `Prompt` maintains a node registry. Reading the same coordinate always
returns the **identical** `ActivationRef` object — no duplicates are created:

```python
>>> ref1 = p[0][5]["resid_post"]
>>> ref2 = p[0][5]["resid_post"]
>>> ref1 is ref2
True
```

### Example: Reading an activation reference

```python
>>> p = prompts[0]
>>> p[0][5]["resid_post"]
Ref(P0.TIndexSelector(0).LIndexSelector(5).resid_post[deps=0])
```

The `deps=N` suffix shows how many causal writes were captured in the
dependency snapshot at instantiation time (see below).

### Example: Patching an activation

```python
>>> p = prompts[0]
>>> q = prompts[1]

# Patch p's token-3 / layer-5 / resid_post with the activation from q
>>> p[3][5]["resid_post"] = q[2][5]["resid_post"]

# Generate with the patch applied
>>> generate(p)
```

### Arithmetic on activations

You can perform arithmetic operations on activation references:

```python
>>> p[0][5]["resid_post"] = q[0][5]["resid_post"] * 2.0
>>> p[0][5]["resid_post"] = q[0][5]["resid_post"] + r[0][5]["resid_post"]
>>> p[0][5]["resid_post"] = (q[0][5]["resid_post"] - r[0][5]["resid_post"]) * 0.5
```

Supported operations: `+`, `-`, `*`, `/`, `**`, `@` and all other PyTorch
tensor operations.

### Modification ledger and dependency snapshots

**Writes** (`p[tok][layer][module] = value`) are recorded in `p`'s
modification ledger.  They become active patches during the next forward
pass.  The latest write to a given position supersedes earlier ones.

**Reads** (`p[tok][layer][module]`) instantiate an `ActivationRef` whose
**dependency snapshot** is frozen at creation time.  The snapshot contains
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
>>> ref = q[2][5]["resid_post"]   # snapshot captures original_value
>>> q[1][3]["resid_post"] = new_value   # too late — ref already frozen
>>> p[3][5]["resid_post"] = ref
>>> generate(p)   # uses original_value for q's activation, not new_value
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
| **ESC** | Switch between COMMAND and INSTRUCT modes |
| **Enter** | Execute command / Store prompt |
| **Up/Down** | Navigate command history |
| **Ctrl+C** | Exit the REPL |
| **q** / **quit** | Exit (in COMMAND mode) |

---

## Example Session

```
Enter Model ID (default: Qwen/Qwen3-0.6B): 
[-] Loading model: Qwen/Qwen3-0.6B...
[+] Model loaded successfully on cpu

Starting in COMMAND mode. Press ESC to switch modes.
Type 'help' in COMMAND mode for available commands.

>>> # Press ESC to switch to INSTRUCT mode

> The capital of Australia is
The capital of Australia is
·  ·       ·  ·         ·

> The capital of Austria is
The capital of Austria is
·  ·       ·  ·         ·

>>> # Press ESC to switch back to COMMAND mode

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
>>> p1[3][0]["resid_post"] = p0[3][0]["resid_post"]

>>> generate(p1)
[Dependency] P1 needs -> P0:T0
[Execute] Extracting 1 values from P0...
[Execute] Running P1 with 1 patches...
'The capital of Austria is the city of Darwin...'  # Patched!
```

---

## Module Reference

### witt (Core Library)

| Module | Contents |
|--------|----------|
| `witt.prompt` | `Prompt`, `PromptList`, `TokenProxy`, `LayerProxy` |
| `witt.computational_node` | `ComputationalNode`, `ActivationRef`, `ConstantNode`, `WriteRecord` |
| `witt.state_node` | `StateNode` — Retained for compatibility |
| `witt.load` | `load_model()`, `load_tokenizer()` |
| `witt.tokenize` | `tokenize()` |

### env (Execution Environment)

| Module | Contents |
|--------|----------|
| `env.environment` | `ExecutionEnvironment` — REPL runtime |

### ui (User Interface)

| Module | Contents |
|--------|----------|
| `ui.cli` | `run()` — Entry point |
| `ui.input_processor` | `InputProcessor` — REPL orchestration |
| `ui.inspector` | `PromptInspector`, `InspectResult`, `LiveInspectDisplay` |
| `ui.screen_buffer` | `ScreenBuffer` — Terminal display management |

---

## License

See [LICENSE](LICENSE) for details.

