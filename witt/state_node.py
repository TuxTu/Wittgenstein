from typing import Optional, Tuple


class StateNode:
    """
    A node in the intervention tree.
    Represents the state of a specific prompt at a specific time.
    """
    def __init__(self, prompt_index: int, parent: Optional['StateNode'] = None, 
                 patch_target=None, patch_value_node=None):
        self.prompt_index = prompt_index
        self.parent = parent  # The previous version of THIS prompt
        self.patch_target = patch_target
        self.patch_value_node = patch_value_node
        
        # Calculate depth (time) implicitly
        self.time_step = (parent.time_step + 1) if parent else 0

    @property
    def key(self) -> Tuple[int, int]:
        """The unique composite key for this node."""
        return (self.prompt_index, self.time_step)

    def __repr__(self):
        if self.parent is None:
            return f"P{self.prompt_index}|t={self.time_step}:Clean"
        return f"P{self.prompt_index}:t={self.time_step}"

