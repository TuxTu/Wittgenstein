"""
Core execution engine for model inference with activation patching and extraction.

This module provides the Executor class which handles:
- Running model forward/generate passes
- Applying activation patches (interventions)
- Extracting activation values from specific locations
- Resolving dependencies between computational nodes
"""
import torch
from typing import List, Optional, Callable
from collections import defaultdict

from .state_node import StateNode
from .computational_node import ComputationalNode, ActivationRef, BinaryOpNode


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
    
    @staticmethod
    def collect_leaves(node: ComputationalNode) -> List[ActivationRef]:
        """
        Recursively collect all unfilled ActivationRef leaves from a computational graph.
        
        Args:
            node: The root of the computational graph to traverse
            
        Returns:
            List of ActivationRef nodes that haven't been evaluated yet
        """
        leaves = []
        if isinstance(node, ActivationRef) and node.evaluate() is None:
            leaves.append(node)
        elif isinstance(node, BinaryOpNode) and node.evaluate() is None:
            leaves.extend(Executor.collect_leaves(node.left))
            leaves.extend(Executor.collect_leaves(node.right))
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
        max_new_tokens,
        mode: str = "inference", 
        extraction_targets: Optional[List[ActivationRef]] = None
    ) -> Optional[str]:
        """
        A unified runner for model execution with patching and extraction.
        
        Args:
            state: The StateNode representing the current prompt state
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
                
                # A. Apply Injections (patches)
                if is_prefill:
                    for patch_node in history_patches:
                        t_layer, t_token, t_mod = patch_node.patch_target
                        if t_layer == layer_idx and t_mod == module_name:
                            if t_token < tensor.shape[1]:
                                val = patch_node.patch_value_node.evaluate()
                                val = val.to(tensor.device).to(tensor.dtype)
                                tensor[:, t_token, :] = val
                
                # B. Apply Extractions
                if mode == "extraction" and extraction_targets:
                    if is_prefill:
                        for ref in extraction_targets:
                            if ref.layer_idx == layer_idx and ref.module == module_name:
                                if ref.token_idx < tensor.shape[1]:
                                    data = tensor[:, ref.token_idx, :].clone().detach()
                                    ref.set_cache(data)
                
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
        
        # Collect locations from patches
        for p in history_patches:
            t_layer, _, t_mod = p.patch_target
            needed_hooks.add((t_layer, t_mod))
            
        # Collect locations from extractions
        if mode == "extraction" and extraction_targets:
            for ref in extraction_targets:
                needed_hooks.add((ref.layer_idx, ref.module))
                
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
    
    def generate(self, prompt, max_new_tokens: int = 1024) -> str:
        """
        Generate text for a prompt, applying any patches in its history.
        
        Args:
            prompt: A Prompt object
            
        Returns:
            The generated text
        """
        return self.execute_pass(prompt.head, mode="inference", max_new_tokens=max_new_tokens)

