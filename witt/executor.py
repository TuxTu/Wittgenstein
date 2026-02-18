"""
Core execution engine for model inference with activation patching and extraction.

This module provides the Executor class which handles:
- Running model forward/generate passes
- Applying activation patches (interventions)
- Extracting activation values from specific locations
- Resolving dependencies between computational nodes

Supports Selector-based token and layer addressing for both
extraction and injection.
"""
import torch
from typing import List, Optional, Callable
from collections import defaultdict

from .state_node import StateNode
from .computational_node import ComputationalNode, ActivationRef


class Executor:
    """
    Core execution engine for running model inference with activation interventions.
    
    The Executor handles:
    - Forward passes with activation patching
    - Extraction of activation values at specific locations
    - Automatic resolution of dependencies between prompts
    
    Args:
        model: The HuggingFace transformer model
        tokenizer: The corresponding tokenizer
        prompts: A PromptList containing all prompts
    """
    
    def __init__(self, model, tokenizer, prompts):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = prompts

    @property
    def num_layers(self) -> int:
        """Return the number of transformer layers in the model."""
        base_model = getattr(self.model, "model", None) or getattr(self.model, "transformer", None)
        if not base_model:
            base_model = self.model
        layers = getattr(base_model, "layers", None) or getattr(base_model, "h", None)
        if not layers:
            raise ValueError(f"Could not locate layers in model {type(self.model)}")
        return len(layers)
    
    @staticmethod
    def collect_leaves(node: ComputationalNode) -> List[ActivationRef]:
        """
        Recursively collect all unfilled ActivationRef leaves from a computational graph.
        
        Uses the generic ``children()`` interface so it works with all node types
        (BinaryOpNode, TorchFunctionNode, MethodCallNode, etc.) without
        type-specific checks.
        
        Args:
            node: The root of the computational graph to traverse
            
        Returns:
            List of ActivationRef nodes that haven't been evaluated yet
        """
        if isinstance(node, ActivationRef) and node.evaluate() is None:
            return [node]
        leaves = []
        for child in node.children():
            leaves.extend(Executor.collect_leaves(child))
        return leaves
    
    def get_transformer_module(self, layer_idx: int, module_name: str):
        """
        Find the specific PyTorch module for a hook at the given layer and location.
        
        Supports common transformer architectures (LLaMA, GPT, etc.)
        
        Args:
            layer_idx: The layer index (0-based)
            module_name: One of "resid_pre", "resid_post", "mlp", or "attn"
            
        Returns:
            The PyTorch module, or None if not found
        """
        # Find the list of layers (handle different architectures)
        base_model = getattr(self.model, "model", None) or getattr(self.model, "transformer", None)
        if not base_model:
            base_model = self.model  # Fallback
            
        layers = getattr(base_model, "layers", None) or getattr(base_model, "h", None)
        if not layers:
            raise ValueError(f"Could not locate layers in model {type(self.model)}")
            
        if layer_idx >= len(layers):
            raise ValueError(f"Layer index {layer_idx} out of bounds")
            
        layer = layers[layer_idx]
        
        # Find the sub-module based on module_name
        if module_name in ["resid_pre", "resid_post"]:
            return layer
        elif module_name == "mlp":
            return getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None)
        elif module_name == "attn":
            return getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
            
        return None
    
    def resolve_dependencies(self, state: StateNode):
        """
        Look at the AST, find ActivationRefs, group them by (Prompt, Node_id),
        and recursively execute to fill in their values.
        
        Args:
            state: The StateNode whose patch_value_node dependencies need resolving
        """
        if not state.patch_value_node:
            return

        # Collect all leaves (ActivationRefs that need values)
        needed_refs = self.collect_leaves(state.patch_value_node)

        # Group by Coordinate Key: (PromptID, TimeStep)
        refs_by_coordinate = defaultdict(list)
        for ref in needed_refs:
            # ref.key returns (prompt_id, time_step)
            refs_by_coordinate[ref.key].append(ref)

        # Iterate and solve each dependency
        for (p_idx, t_step), refs_to_fill in refs_by_coordinate.items():
            # Find the prompt object
            if p_idx >= len(self.prompts):
                raise ValueError(f"Dependency refers to non-existent Prompt {p_idx}")
                
            dependency_state = self.prompts[p_idx].get_state_at(t_step)
            
            print(f"[Dependency] P{state.prompt_index} needs -> P{p_idx}:T{t_step}")

            # RECURSION: execute the dependency to fill its values
            self.execute_pass(
                dependency_state, 
                mode="extraction", 
                extraction_targets=refs_to_fill
            )
    
    def execute_pass(
        self, 
        state: StateNode, 
        max_new_tokens=None,
        mode: str = "inference", 
        extraction_targets: Optional[List[ActivationRef]] = None
    ) -> Optional[str]:
        """
        A unified runner for model execution with patching and extraction.
        
        Args:
            state: The StateNode representing the current prompt state
            max_new_tokens: Maximum tokens to generate (inference mode)
            mode: Either "inference" (generate output) or "extraction" (extract activations)
            extraction_targets: List of ActivationRef nodes to fill (for extraction mode)
            
        Returns:
            Generated text if mode="inference", None if mode="extraction"
        """
        # First, RECURSIVELY resolve dependencies for this state
        self.resolve_dependencies(state)

        # Collect patches from the state history
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

        # Prepare Input IDs
        token_ids_list = self.prompts[state.prompt_index].token_ids
        input_ids = torch.tensor(token_ids_list, device=self.model.device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        prompt_len = input_ids.shape[1]
        n_layers = self.num_layers

        # Define the Unified Hook Factory
        def create_hook(layer_idx: int, module_name: str, is_pre: bool = False) -> Callable:
            def hook(module, args, output):
                # Get the tensor
                if is_pre:
                    tensor = args[0]
                else:
                    tensor = output[0] if isinstance(output, tuple) else output
                
                # Check if we are in the "Prefill" phase (processing the prompt)
                is_prefill = (tensor.shape[1] == prompt_len)
                
                seq_len = tensor.shape[1]

                # A. Apply Injections (patches)
                if is_prefill:
                    for patch_node in history_patches:
                        p_layer_sel, p_token_sel, p_mod = patch_node.patch_target
                        if p_mod != module_name:
                            continue
                        # Check if this layer is in the patch's layer selector
                        patch_layer_indices = p_layer_sel.indices(n_layers)
                        if layer_idx not in patch_layer_indices:
                            continue
                        # Resolve token selector
                        tok_idx = p_token_sel.resolve(seq_len)
                        val = patch_node.patch_value_node.evaluate()
                        if val is not None:
                            val = val.to(tensor.device).to(tensor.dtype)
                            tensor[:, tok_idx, :] = val
                
                # B. Apply Extractions
                if mode == "extraction" and extraction_targets:
                    if is_prefill:
                        for ref in extraction_targets:
                            if ref.module != module_name:
                                continue
                            # Check if this layer is in the ref's layer selector
                            ref_layer_indices = ref.layer_selector.indices(n_layers)
                            if layer_idx not in ref_layer_indices:
                                continue
                            # Resolve token selector and extract
                            tok_idx = ref.token_selector.resolve(seq_len)
                            data = tensor[:, tok_idx, :].clone().detach()
                            if ref.layer_selector.is_single:
                                ref.set_cache(data)
                            else:
                                ref.set_layer_cache(layer_idx, data)
                
                # Return modified tensors
                if is_pre:
                    return (tensor,) + args[1:]
                else:
                    if isinstance(output, tuple):
                        return (tensor,) + output[1:]
                    else:
                        return tensor
            return hook

        # Register Hooks
        hook_handles = []
        needed_hooks = set()
        
        # Collect locations from patches (now with selectors)
        for p in history_patches:
            p_layer_sel, _, p_mod = p.patch_target
            for li in p_layer_sel.indices(n_layers):
                needed_hooks.add((li, p_mod))
            
        # Collect locations from extractions (now with selectors)
        if mode == "extraction" and extraction_targets:
            for ref in extraction_targets:
                for li in ref.layer_selector.indices(n_layers):
                    needed_hooks.add((li, ref.module))
                
        # Attach hooks
        for layer_idx, module_name in needed_hooks:
            mod = self.get_transformer_module(layer_idx, module_name)
            if mod is None:
                continue
                
            is_pre = (module_name == "resid_pre")
            if is_pre:
                h = mod.register_forward_pre_hook(create_hook(layer_idx, module_name, is_pre=True))
            else:
                h = mod.register_forward_hook(create_hook(layer_idx, module_name, is_pre=False))
            hook_handles.append(h)

        # Run Model
        try:
            if mode == "inference":
                # Generate
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                output_ids = self.model.generate(
                    input_ids, 
                    max_new_tokens=max_new_tokens,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id
                )
                print(self.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=False))
                return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            else:
                # Forward pass only (for extraction)
                with torch.no_grad():
                    self.model(input_ids, attention_mask=attention_mask)
                return None
        finally:
            # Always clean up hooks
            for h in hook_handles:
                h.remove()
    
    def generate(self, prompt, max_new_tokens: int = 35536) -> str:
        """
        Generate text for a prompt, applying any patches in its history.
        
        Args:
            prompt: A Prompt object
            
        Returns:
            The generated text
        """
        return self.execute_pass(prompt.head, mode="inference", max_new_tokens=max_new_tokens)
