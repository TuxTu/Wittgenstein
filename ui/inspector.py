import shutil
import signal
import sys
from typing import List, Tuple, Optional

from .screen_buffer import ScreenBuffer


class InspectResult:
    """
    Stores tokenization result and re-renders dynamically based on terminal width.
    
    Every time this object is displayed (via repr), it recalculates the layout
    using the current terminal width.
    """
    
    def __init__(self, prompt: str, token_seq: List[Tuple[int, str]]):
        self.prompt = prompt
        self.token_seq = token_seq
        
        # Pre-compute display tokens and indices
        self._tokens = []
        new_line = '\n'
        tab = '\t'
        for (token_id, token_str) in token_seq:
            display_token = token_str.replace('Ġ', ' ').replace('Ċ', new_line).replace('ĉ', tab)
            self._tokens.append((display_token, token_id))
        
        # Track last render info for redrawing
        self._last_line_count = 0
    
    def render(self, terminal_width: int = None) -> str:
        """
        Render the tokenization with the given terminal width.
        
        Args:
            terminal_width: Width to use (None = auto-detect current terminal)
        
        Returns:
            Formatted string with tokens and indices
        """
        if terminal_width is None:
            terminal_width = shutil.get_terminal_size().columns
        
        # Build list of (display_token, index_str, column_width)
        tokens_with_width = []
        for display_token, token_id in self._tokens:
            width = len(display_token)
            tokens_with_width.append((display_token, width))
        
        # Group tokens into lines based on terminal width
        lines = []
        current_token_line = ""
        current_index_line = ""
        current_width = 0
        
        for display_token, width in tokens_with_width:
            # Check if adding this token would exceed terminal width
            if current_width + width > terminal_width and current_width > 0:
                lines.append((current_token_line.rstrip(), current_index_line.rstrip()))
                current_token_line = ""
                current_index_line = ""
                current_width = 0
            
            current_token_line += display_token.ljust(width)
            current_index_line += "·".ljust(width)
            # current_index_line += "^".ljust(width)
            current_width += width
        
        # Don't forget the last line
        if current_token_line:
            lines.append((current_token_line.rstrip(), current_index_line.rstrip()))
        
        # Build output string
        output_lines = []
        for token_line, index_line in lines:
            output_lines.append(token_line)
            output_lines.append(index_line)
        
        # Track line count for redrawing
        self._last_line_count = len(output_lines)
        
        # Use \r\n for compatibility with raw terminal mode
        return "\r\n".join(output_lines)
    
    def __repr__(self) -> str:
        """Re-render with current terminal width when displayed."""
        return self.render() + "\r\n"
    
    def __len__(self) -> int:
        """Number of tokens."""
        return len(self._tokens)
    
    def __getitem__(self, idx):
        """Access individual tokens: (display_token, index, token_id)."""
        return self._tokens[idx]


class LiveInspectDisplay:
    """
    Manages live redrawing of InspectResult on terminal resize.
    Uses ScreenBuffer for complete screen management.
    """
    
    _instance: Optional["LiveInspectDisplay"] = None
    
    def __init__(self):
        self._current_result: Optional[InspectResult] = None
        self._line_count = 0
        self._terminal_width = 0
        self._active = False
        self._original_handler = None
        self._prompt_provider = None
        self._screen_buffer = ScreenBuffer.get()
    
    @classmethod
    def get(cls) -> "LiveInspectDisplay":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def set_prompt_provider(self, provider):
        """
        Set a callback to get current prompt state.
        
        The provider should return a tuple:
            (prompt_text, buffer_list, cursor_idx) or None if no active prompt
        """
        self._prompt_provider = provider
    
    def _calculate_line_count(self, content_length: int, terminal_width: int) -> int:
        """Calculate how many terminal lines content occupies."""
        if terminal_width <= 0:
            return 1
        return max(1, (content_length + terminal_width - 1) // terminal_width)
    
    def _get_cursor_position(self) -> Tuple[int, int]:
        """
        Query terminal for current cursor position.
        Returns (row, col) or (0, 0) if unable to query.
        """
        try:
            import termios
            import tty
            
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            
            try:
                tty.setraw(fd)
                sys.stdout.write('\x1b[6n')  # Query cursor position
                sys.stdout.flush()
                
                response = ''
                while True:
                    ch = sys.stdin.read(1)
                    response += ch
                    if ch == 'R':
                        break
                    if len(response) > 20:  # Safety limit
                        break
                
                # Response format: \x1b[{row};{col}R
                if response.startswith('\x1b[') and response.endswith('R'):
                    parts = response[2:-1].split(';')
                    if len(parts) == 2:
                        return int(parts[0]), int(parts[1])
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass
        
        return 0, 0
    
    def display(self, result: InspectResult):
        """Display a result and track it for live updates."""
        self._current_result = result
        self._terminal_width = shutil.get_terminal_size().columns
        
        # Set up dynamic content renderer for ScreenBuffer
        self._screen_buffer.set_dynamic_content(lambda: result.render())
        
        # Print and track line count
        output = repr(result)
        print(output, end='')
        self._line_count = result._last_line_count
        
        # Enable live updates
        self.enable_signal()
    
    def enable_signal(self):
        """Enable SIGWINCH handler for resize detection."""
        if self._active:
            return
        
        try:
            self._original_handler = signal.signal(signal.SIGWINCH, self._on_resize)
            self._active = True
        except (ValueError, OSError):
            # Signal handling not available (e.g., not main thread)
            pass
    
    def disable(self):
        """Disable live updates and restore original handler."""
        if self._active and self._original_handler is not None:
            try:
                signal.signal(signal.SIGWINCH, self._original_handler)
            except (ValueError, OSError):
                pass
        self._active = False
        self._current_result = None
    
    def _on_resize(self, signum, frame):
        """Handle terminal resize by redrawing entire screen via ScreenBuffer."""
        if not self._screen_buffer._active:
            return
        
        try:
            # Get current prompt state if available
            prompt_state = None
            if self._prompt_provider:
                try:
                    prompt_state = self._prompt_provider()
                except Exception:
                    prompt_state = None
            
            # Let ScreenBuffer handle the complete redraw
            self._screen_buffer.redraw(prompt_state)
            
            # Update tracking
            self._terminal_width = shutil.get_terminal_size().columns
            if self._current_result:
                self._line_count = self._current_result._last_line_count
            
        except Exception:
            # Silently fail if redraw doesn't work
            pass


class PromptInspector:
    """
    Encapsulates the tokenizer-backed inspection of a prompt.
    """
    def __init__(self, model_id, live_resize=True):
        """
        Args:
            model_id: The model/tokenizer to load
            live_resize: If True, results redraw automatically on terminal resize
        """
        try:
            print(f"\n\r[-] Initializing inspector for {model_id}...")
        except Exception as e:
            print(f"\n\r[!] Critical Error: {e}")
            raise
        
        self.live_resize = live_resize
        self._live_display = LiveInspectDisplay.get() if live_resize else None

    def inspect(self, prompt_item, live=None) -> InspectResult:
        """
        Tokenizes the supplied prompt and returns an InspectResult.
        
        The result dynamically re-renders based on terminal width whenever displayed.
        
        Args:
            prompt_item: An object with .text and .tokens attributes (e.g. Prompt)
            live: Override live_resize setting for this call (None = use default)
            
        Returns:
            InspectResult that can be displayed multiple times with different widths
        """
        result = InspectResult(prompt_item.text, prompt_item.tokens)
        
        # Determine if we should use live display
        use_live = live if live is not None else self.live_resize
        
        if use_live and self._live_display:
            # Use live display (tracks for resize updates)
            self._live_display.display(result)
        else:
            # Simple print
            print(result)
        
        return result
    
    def stop_live(self):
        """Stop live resize updates."""
        if self._live_display:
            self._live_display.disable()
