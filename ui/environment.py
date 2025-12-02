"""
Execution environment for the REPL.
"""
import sys
import io
import torch
from typing import List, Any, Dict, Set
from collections import defaultdict

from llm_cores import Prompt, PromptList, load_model, load_tokenizer, tokenize
from llm_cores.state_node import StateNode
from llm_cores.computational_node import ComputationalNode, ActivationRef, BinaryOpNode


class HelpDisplay:
    """A helper that displays help when repr'd (so typing 'help' shows help)."""
    
    def __init__(self, help_func):
        self._help_func = help_func
    
    def __repr__(self):
        self._help_func()
        return ""
    
    def __call__(self):
        self._help_func()


class ProtectedNamespace(dict):
    """
    A dict subclass that prevents modification of protected keys.
    """
    
    def __init__(self, protected_keys: Set[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._protected_keys = protected_keys
        self._locked = False
    
    def lock(self):
        """Lock the namespace to prevent modification of protected keys."""
        self._locked = True
    
    def __setitem__(self, key, value):
        if self._locked and key in self._protected_keys:
            raise NameError(f"Cannot reassign protected variable '{key}'")
        super().__setitem__(key, value)
    
    def __delitem__(self, key):
        if self._locked and key in self._protected_keys:
            raise NameError(f"Cannot delete protected variable '{key}'")
        super().__delitem__(key)
    
    def update(self, *args, **kwargs):
        # When updating, check if any protected keys are being modified
        if self._locked:
            other = dict(*args, **kwargs)
            for key in other:
                if key in self._protected_keys:
                    raise NameError(f"Cannot reassign protected variable '{key}'")
        super().update(*args, **kwargs)


class ExecutionEnvironment:
    """
    Maintains a persistent execution context for COMMAND mode.
    
    Features:
    - Persistent namespace across commands (variables survive between executions)
    - Access to stored prompts via `prompts` variable
    - Access to inspector via `inspect()` function
    - Command history tracking
    """
    
    # Keys that cannot be reassigned by user code
    PROTECTED_KEYS = {
        'prompts', 'env', 'inspect', 'inspector', 'last', 'help', 'struct',
    }
    
    def __init__(self, inspector, model_id):
        self.inspector = inspector
        self.model_id = model_id
        self.tokenizer = load_tokenizer(self.model_id)
        self.model = load_model(self.model_id)
        self.prompts = PromptList()
        
        # The persistent namespace for exec() with protected keys
        self._namespace = ProtectedNamespace(self.PROTECTED_KEYS)
        
        # Expose built-in utilities in the namespace
        self._setup_namespace()
        
        # Lock the namespace after setup
        self._namespace.lock()

        # Is silent result?
        self._is_silent_result = False
    
    def _setup_namespace(self):
        """Initialize the namespace with useful bindings."""
        self._namespace.update({
            # Core objects
            'prompts': self.prompts,
            'env': self,
            
            # Inspector access
            'inspect': self._inspect_wrapper,
            'inspector': self.inspector,
            
            # Model access
            'model': self.model,
            'tokenizer': self.tokenizer,

            # Generation access
            'generate': self._generate_wrapper,
            
            # Convenience functions
            'last': lambda: self.prompts.last,
            'help': HelpDisplay(self._show_help),
            'struct': HelpDisplay(self._show_structure),
        })

    def _show_structure(self):
        """Print the structure of the selected model."""
        print(str(self.model))
    
    def _inspect_wrapper(self, text_or_prompt):
        """
        Wrapper for inspector.inspect that accepts either a string or Prompt object.
        If a Prompt is passed, the result is also stored in prompt.result.
        """
        self._is_silent_result = True
        if isinstance(text_or_prompt, Prompt):
            result = self.inspector.inspect(text_or_prompt)
            text_or_prompt.result = result
            return result
        elif isinstance(text_or_prompt, PromptList):
            results = ""
            for prompt in text_or_prompt:
                result = self.inspector.inspect(prompt)
                results += repr(result)
            return results
    
    @staticmethod
    def _collect_leaves(node: ComputationalNode) -> List[ActivationRef]:
        leaves = []
        if isinstance(node, ActivationRef) and node.evaluate() is None:
            leaves.append(node)
        elif isinstance(node, BinaryOpNode) and node.evaluate() is None:
            leaves.extend(ExecutionEnvironment._collect_leaves(node.left))
            leaves.extend(ExecutionEnvironment._collect_leaves(node.right))
        return leaves

    def _get_transformer_module(self, layer_idx: int, module_name: str):
        """Helper to find the specific PyTorch module for a hook."""
        # 1. Find the list of layers
        base_model = getattr(self.model, "model", None) or getattr(self.model, "transformer", None)
        if not base_model:
            base_model = self.model # Fallback
            
        layers = getattr(base_model, "layers", None) or getattr(base_model, "h", None)
        if not layers:
            raise ValueError(f"Could not locate layers in model {type(self.model)}")
            
        if layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of bounds")
            
        layer = layers[layer_idx]
        
        # 2. Find the sub-module
        if module_name in ["resid_pre", "resid_post"]:
            return layer
        elif module_name == "mlp":
            return getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        elif module_name == "attn":
            return getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
            
        return None

    def _resolve_dependencies(self, state: StateNode):
        """
        Looks at the AST, finds Refs, groups them by (Prompt, Node_id), 
        and recursively executes them.
        """
        if not state.patch_value_node:
            return

        # 1. Collect all leaves (ActivationRefs)
        needed_refs = self._collect_leaves(state.patch_value_node)

        # 2. Group by Coordinate Key: (PromptID, TimeStep)
        refs_by_coordinate = defaultdict(list)
        for ref in needed_refs:
            # ref.key returns (prompt_id, time_step)
            refs_by_coordinate[ref.key].append(ref)

        # 3. Iterate and Solve
        for (p_idx, t_step), refs_to_fill in refs_by_coordinate.items():
            # Find the prompt object
            if p_idx >= len(self.prompts):
                raise ValueError(f"Dependency refers to non-existent Prompt {p_idx}")
                
            dependency_state = self.prompts[p_idx].get_state_at(t_step)
            
            print(f"[Dependency] P{state.prompt_index} needs -> P{p_idx}:T{t_step}")

            # RECURSION
            self._execute_pass(
                dependency_state, 
                mode="extraction", 
                extraction_targets=refs_to_fill
            )

    def _execute_pass(self, state: StateNode, mode: str = "inference", extraction_targets: List[ActivationRef] = None):
        """
        A unified runner.
        mode="inference": Apply patches, return generation.
        mode="extraction": Apply patches (if any), fill 'extraction_targets', return None.
        """
        
        # 1. First, RECURSIVELY resolve dependencies for *this* state.
        self._resolve_dependencies(state)

        # 2. Identify Patches
        history_patches = []
        curr = state
        while curr.parent is not None:
            if curr.patch_value_node:
                history_patches.append(curr)
            curr = curr.parent
            
        if history_patches:
            print(f"[Execute] Running P{state.prompt_index} with {len(history_patches)} patches...")

        if mode == "extraction" and extraction_targets:
            print(f"[Execute] Extracting {len(extraction_targets)} values from P{state.prompt_index}...")

        # Prepare Input IDs (needed for hook logic)
        token_ids_list = self.prompts[state.prompt_index].token_ids
        input_ids = torch.tensor(token_ids_list, device=self.model.device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        prompt_len = input_ids.shape[1]

        # 3. Define the Unified Hook
        def create_hook(layer_idx, module_name, is_pre=False):
            def hook(module, args, output):
                # Get the tensor
                if is_pre:
                    tensor = args[0]
                else:
                    tensor = output[0] if isinstance(output, tuple) else output
                
                # Check if we are in the "Prefill" phase (processing the prompt)
                # If tensor length matches prompt length, we are processing the prompt.
                # If tensor length is 1 (and prompt > 1), we are in decoding phase.
                # We only apply prompt patches during the prefill phase.
                is_prefill = (tensor.shape[1] == prompt_len)
                
                # A. Apply Injections
                if is_prefill:
                    for patch_node in history_patches:
                        t_layer, t_token, t_mod = patch_node.patch_target
                        if t_layer == layer_idx and t_mod == module_name:
                            # Verify token index is valid for this tensor
                            if t_token < tensor.shape[1]:
                                val = patch_node.patch_value_node.evaluate()
                                # Ensure device/dtype match
                                val = val.to(tensor.device).to(tensor.dtype)
                                tensor[:, t_token, :] = val
                
                # B. Apply Extractions
                if mode == "extraction" and extraction_targets:
                    # Extraction usually happens on the prompt tokens (prefill)
                    if is_prefill:
                        for ref in extraction_targets:
                            if ref.layer_idx == layer_idx and ref.module == module_name:
                                if ref.token_idx < tensor.shape[1]:
                                    data = tensor[:, ref.token_idx, :].clone().detach()
                                    ref.set_cache(data)
                
                # Return
                if is_pre:
                    return (tensor,) + args[1:]
                else:
                    if isinstance(output, tuple):
                        return (tensor,) + output[1:]
                    else:
                        return tensor
            return hook

        # 4. Register Hooks
        hook_handles = []
        needed_hooks = set()
        
        # Collect locations from patches
        for p in history_patches:
            t_layer, _, t_mod = p.patch_target
            needed_hooks.add((t_layer, t_mod))
            
        # Collect locations from extractions
        if mode == "extraction" and extraction_targets:
            for ref in extraction_targets:
                needed_hooks.add((ref.layer_idx, ref.module))
                
        # Attach
        for layer_idx, module_name in needed_hooks:
            mod = self._get_transformer_module(layer_idx, module_name)
            if mod is None:
                continue
                
            is_pre = (module_name == "resid_pre")
            if is_pre:
                h = mod.register_forward_pre_hook(create_hook(layer_idx, module_name, is_pre=True))
            else:
                h = mod.register_forward_hook(create_hook(layer_idx, module_name, is_pre=False))
            hook_handles.append(h)

        # 5. Run Model
        try:
            if mode == "inference":
                # Generate
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                output_ids = self.model.generate(
                    input_ids, 
                    attention_mask=attention_mask,
                    max_new_tokens=20,
                    pad_token_id=pad_token_id
                )
                return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            else:
                # Forward pass only
                with torch.no_grad():
                    self.model(input_ids, attention_mask=attention_mask)
                return None
        finally:
            for h in hook_handles:
                h.remove()


    def _generate_wrapper(self, prompt: Prompt):
        """
        Wrapper for model.generate that accepts a Prompt object.
        """
        return self._execute_pass(prompt.head, mode="inference")

    def execute(self, code: str) -> str:
        """
        Execute code in the persistent namespace.
        
        Captures all stdout output (including print() calls) during execution.
        
        Args:
            code: Python code to execute
            
        Returns:
            All output produced during execution as a string
        """
        # Capture stdout during execution
        old_stdout = sys.stdout
        captured = io.StringIO()
        sys.stdout = captured
        
        self._is_silent_result = False

        try:
            # Try to evaluate as expression first (for things like `x + 1`)
            try:
                result = eval(code, self._namespace)
                if result is not None:
                    if not self._is_silent_result:
                        print(repr(result))
            except SyntaxError:
                # Not an expression, execute as statement
                exec(code, self._namespace)
        finally:
            sys.stdout = old_stdout
        
        return captured.getvalue()
    
    def add_prompt(self, text: str) -> Prompt:
        """Store a prompt from INSTRUCT mode."""
        tokens = tokenize(self.tokenizer, text)
        return self.prompts.add(text, tokens)
    
    def get_variable(self, name: str) -> Any:
        """Get a variable from the namespace."""
        return self._namespace.get(name)
    
    def set_variable(self, name: str, value: Any):
        """Set a variable in the namespace (protected keys cannot be set)."""
        if name in self.PROTECTED_KEYS:
            raise NameError(f"Cannot reassign protected variable '{name}'")
        self._namespace[name] = value
    
    @property
    def variables(self) -> Dict[str, Any]:
        """Get all user-defined variables (excluding built-ins)."""
        return {k: v for k, v in self._namespace.items() if k not in self.PROTECTED_KEYS}
    
    def clear(self):
        """Clear all user-defined variables (keeps built-ins)."""
        # Temporarily unlock to clear and reset
        self._namespace._locked = False
        self._namespace.clear()
        self._setup_namespace()
        self._namespace.lock()
    
    def _show_help(self):
        """Print help about available commands and variables."""
        help_lines = [
            "=== COMMAND MODE HELP ===",
            "",
            "Built-in Variables:",
            "  prompts    - All stored INSTRUCT prompts (PromptList)",
            "  env        - The execution environment",
            "  inspector  - The PromptInspector instance",
            "",
            "Built-in Functions:",
            "  inspect(p) - Inspect a prompt (string or Prompt object)",
            "  last()     - Get the most recent prompt",
            "  struct     - Display model structure",
            "",
            "Prompt Access:",
            "  prompts[0]    - First prompt",
            "  prompts[-1]   - Last prompt",
            "  prompts.last  - Most recent prompt",
            "  len(prompts)  - Number of stored prompts",
            "",
            "Prompt Properties:",
            "  p.text     - The prompt text",
            "  p.index    - Sequential index",
            "  p.tag(...) - Add tags to a prompt",
            "",
            "Commands:",
            "  q   - Quit",
            "  ESC - Switch modes",
            "",
        ]
        for line in help_lines:
            print(line)

