"""
Execution environment for the REPL.
"""
import sys
import io
import torch
from typing import List, Any, Dict, Set

from llm_cores import Prompt, PromptList, load_model, load_tokenizer, tokenize


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
    
    def _generate_wrapper(self, prompt: Prompt):
        """
        Wrapper for model.generate that accepts a Prompt object.
        """
        input_ids = torch.tensor([prompt.token_ids], device=self.model.device)
        return self.tokenizer.decode(self.model.generate(input_ids)[0], skip_special_tokens=True)

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

