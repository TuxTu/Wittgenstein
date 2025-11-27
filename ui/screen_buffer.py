"""
Screen buffer for alternate screen mode content management.
"""
import shutil
import sys
from typing import List, Tuple, Optional, Callable


class ScreenBuffer:
    """
    Maintains a buffer of screen content for alternate screen mode.
    On resize, can redraw all content appropriately.
    
    Stores content as a list of items, where each item is either:
    - A string (static text)
    - A callable that returns rendered content (for InspectResults)
    """
    
    _instance: Optional["ScreenBuffer"] = None
    
    def __init__(self):
        self._items: List = []  # List of (type, content) tuples
        self._scroll_offset = 0
        self._active = False
    
    @classmethod
    def get(cls) -> "ScreenBuffer":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def activate(self):
        """Activate the screen buffer."""
        self._active = True
        self._items = []
        self._scroll_offset = 0
    
    def deactivate(self):
        """Deactivate the screen buffer."""
        self._active = False
    
    def add_line(self, text: str):
        """Add a line of static content."""
        if self._active:
            # Split by newlines and add each line
            for line in text.split('\n'):
                # Remove \r for clean storage
                clean_line = line.replace('\r', '')
                self._items.append(('static', clean_line))
            # Reset scroll when new content is added
            self._scroll_offset = 0
    
    def add_dynamic(self, renderer: Callable[[], str]):
        """
        Add dynamic content that will be re-rendered on each redraw.
        The renderer should return the content for current terminal width.
        """
        if self._active:
            self._items.append(('dynamic', renderer))
            # Reset scroll when new content is added
            self._scroll_offset = 0
    
    def set_dynamic_content(self, renderer: Optional[Callable[[], str]]):
        """Compatibility method - adds dynamic content."""
        if renderer:
            self.add_dynamic(renderer)
    
    def clear_dynamic_content(self):
        """Compatibility method - no-op since we store all dynamics."""
        pass
    
    def scroll_up(self, lines: int = 3):
        """Scroll up (view older content) by the specified number of lines."""
        if self._active:
            self._scroll_offset += lines
    
    def scroll_down(self, lines: int = 3):
        """Scroll down (view newer content) by the specified number of lines."""
        if self._active:
            self._scroll_offset = max(0, self._scroll_offset - lines)
    
    def reset_scroll(self):
        """Reset scroll to bottom (most recent content)."""
        self._scroll_offset = 0
    
    def redraw(self, prompt_state: Optional[Tuple] = None):
        """
        Redraw all content for current terminal size.
        
        Args:
            prompt_state: Optional (prompt_text, buffer, cursor_idx) for active prompt
        """
        if not self._active:
            return
        
        terminal_width = shutil.get_terminal_size().columns
        terminal_height = shutil.get_terminal_size().lines
        
        # Clear screen and go to top
        sys.stdout.write('\x1b[2J\x1b[H')
        
        # Collect all content lines
        all_lines = []
        
        for item_type, content in self._items:
            if item_type == 'static':
                # Static line - wrap if needed
                if len(content) > terminal_width and terminal_width > 0:
                    for i in range(0, len(content), terminal_width):
                        all_lines.append(content[i:i + terminal_width])
                else:
                    all_lines.append(content)
            elif item_type == 'dynamic':
                # Dynamic content - call renderer to get content at current width
                try:
                    rendered = content()  # Call the renderer
                    for line in rendered.split('\r\n'):
                        line = line.replace('\r', '')
                        if len(line) > terminal_width and terminal_width > 0:
                            for i in range(0, len(line), terminal_width):
                                all_lines.append(line[i:i + terminal_width])
                        else:
                            all_lines.append(line)
                except Exception:
                    pass
        
        # Add prompt if provided
        prompt_line = ""
        if prompt_state:
            prompt_text, buffer, cursor_idx = prompt_state
            prompt_line = prompt_text + ''.join(buffer)
        
        # Reserve lines for prompt and scroll indicator
        reserved_lines = 1 if prompt_state else 0
        if self._scroll_offset > 0:
            reserved_lines += 1  # Space for scroll indicator
        
        visible_height = terminal_height - reserved_lines
        total_lines = len(all_lines)
        
        # Handle scrolling with offset
        if total_lines > visible_height:
            # Clamp scroll offset to valid range
            max_scroll = total_lines - visible_height
            self._scroll_offset = max(0, min(self._scroll_offset, max_scroll))
            
            # Calculate window: scroll_offset=0 means bottom, higher means older
            end_idx = total_lines - self._scroll_offset
            start_idx = max(0, end_idx - visible_height)
            visible_lines = all_lines[start_idx:end_idx]
        else:
            visible_lines = all_lines
            self._scroll_offset = 0  # Reset if content fits
        
        # Build all output into a single buffer to avoid flicker
        output_parts = []
        
        # Add visible lines
        output_parts.append('\r\n'.join(visible_lines))
        
        # Add scroll indicator if scrolled up
        if self._scroll_offset > 0:
            indicator = f"── {self._scroll_offset} more lines below ──"
            output_parts.append(f'\r\n\x1b[2m{indicator}\x1b[0m')
        
        # Add prompt on last line
        if prompt_state:
            output_parts.append(f'\r\n{prompt_line}')
        
        # Write everything at once
        sys.stdout.write(''.join(output_parts))
        
        # Move cursor back if needed (must be after the main write)
        if prompt_state:
            chars_after = len(buffer) - cursor_idx
            if chars_after > 0:
                sys.stdout.write(f'\x1b[{chars_after}D')
        
        sys.stdout.flush()
    
    def get_line_count(self) -> int:
        """Get total number of items in buffer."""
        return len(self._items)

