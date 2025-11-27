import sys
import select
import termios
import tty
import os
import shutil

from .inspector import PromptInspector, LiveInspectDisplay
from .screen_buffer import ScreenBuffer
from .environment import ExecutionEnvironment


class InputProcessor:
    """
    Orchestrates the REPL experience, delegating inspection to `PromptInspector`.
    """

    def __init__(self, model_id, default_mode="COMMAND"):
        try:
            self.inspector = PromptInspector(model_id)
        except Exception as exc:
            print(f"\r[!] Failed to initialize prompt inspector: {exc}")
            sys.exit(1)
        
        # Create the execution environment with the inspector
        self.env = ExecutionEnvironment(self.inspector, model_id)
        self.mode = default_mode
        
        # History buffers
        self.history = {
            "COMMAND": [],
            "INSTRUCT": []
        }
        
        # Current prompt state (for live resize)
        self._current_prompt_text = ""
        self._current_buffer = []
        self._current_cursor_idx = 0
        self._in_input = False
        
        # Register as prompt provider for live display
        live_display = LiveInspectDisplay.get()
        live_display.set_prompt_provider(self._get_prompt_state)

    def _get_prompt_state(self):
        """
        Returns current prompt state for live resize.
        Returns (prompt_text, buffer, cursor_idx) or None if not in input.
        """
        if not self._in_input:
            return None
        return (self._current_prompt_text, self._current_buffer, self._current_cursor_idx)
    
    def _update_prompt_state(self, prompt_text, buffer, cursor_idx):
        """Update the tracked prompt state."""
        self._current_prompt_text = prompt_text
        self._current_buffer = list(buffer)  # Copy to avoid reference issues
        self._current_cursor_idx = cursor_idx

    def _getch(self):
        """Read a single character assuming the terminal is already in raw mode."""
        """
        Read a single byte directly from the file descriptor.
        This avoids the buffering issues of sys.stdin.read(1).
        """
        fd = sys.stdin.fileno()
        # Read 1 byte and decode it to string
        return os.read(fd, 1).decode()

    def _visible_len(self, text: str) -> int:
        """Get the visible length of text (excluding control characters like \\r)."""
        return len(text.replace('\r', ''))
    
    def _get_line_count(self, text_len, terminal_width):
        """Calculate how many terminal lines a text of given length occupies."""
        if terminal_width == 0:
            return 1
        return max(1, (text_len + terminal_width - 1) // terminal_width)
    
    def _redraw_line(self, prompt_text, buffer, cursor_idx):
        """
        Clears all lines occupied by the content, prints the new buffer, 
        and places the cursor at the correct index.
        
        Optimized to minimize terminal writes and eliminate blinking.
        """
        full_line = prompt_text + "".join(buffer)
        visible_len = self._visible_len(full_line)
        prompt_visible_len = self._visible_len(prompt_text)
        terminal_width = shutil.get_terminal_size().columns
        
        # Get previous state
        prev_cursor_idx = getattr(self, '_prev_cursor_idx', 0)
        
        # Calculate current cursor position
        if terminal_width > 0:
            prev_cursor_pos = prompt_visible_len + prev_cursor_idx
            current_cursor_line = prev_cursor_pos // terminal_width
        else:
            current_cursor_line = 0
        
        # Build the entire output sequence in memory
        output_parts = []
        
        # 1. Move cursor to the first line
        if current_cursor_line > 0:
            output_parts.append(f'\x1b[{current_cursor_line}A')
        
        # 2. Move to start of line
        output_parts.append('\r')
        
        # 3. Clear from cursor to end of screen (efficient multi-line clear)
        output_parts.append('\x1b[0J')
        
        # 4. Write the new content
        output_parts.append(full_line)
        
        # 5. Position cursor at the correct location
        cursor_pos = prompt_visible_len + cursor_idx
        end_pos = prompt_visible_len + len(buffer)
        
        if terminal_width > 0:
            cursor_line = cursor_pos // terminal_width
            cursor_col = cursor_pos % terminal_width
            end_line = (end_pos - 1) // terminal_width if end_pos > 0 else 0
            
            lines_up = end_line - cursor_line
            if lines_up > 0:
                output_parts.append(f'\x1b[{lines_up}A')
            
            # Move to correct column (1-indexed)
            output_parts.append(f'\x1b[{cursor_col + 1}G')
        else:
            # Fallback for when terminal_width is 0
            chars_after_cursor = len(buffer) - cursor_idx
            if chars_after_cursor > 0:
                output_parts.append(f'\x1b[{chars_after_cursor}D')
        
        # Write everything in a single operation
        sys.stdout.write(''.join(output_parts))
        sys.stdout.flush()
        
        # Update state for next redraw
        self._prev_content_len = visible_len
        self._prev_cursor_idx = cursor_idx
        
        # Update prompt state for live resize
        self._update_prompt_state(prompt_text, buffer, cursor_idx)

    def _read_bracketed_paste(self):
        """
        Read pasted content until the bracketed paste end sequence.
        Returns the pasted text with newlines converted to spaces.
        """
        paste_buffer = []
        while True:
            char = self._getch()
            # Check for end sequence: ESC [ 2 0 1 ~
            if ord(char) == 27:
                # Might be end sequence, check next chars
                seq = char
                for _ in range(5):  # Read [201~
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        seq += self._getch()
                    else:
                        break
                
                if seq == '\x1b[201~':
                    # End of paste
                    break
                else:
                    # Not end sequence, add to buffer
                    paste_buffer.extend(seq)
            elif ord(char) in [10, 13]:
                # Convert newlines to spaces in pasted content
                paste_buffer.append(' ')
            else:
                paste_buffer.append(char)
        
        # Strip trailing spaces (from trailing newlines in copied text)
        return ''.join(paste_buffer).rstrip(' ')

    def _clear_current_input(self):
        """Clear the current multi-line input display."""
        prev_len = getattr(self, '_prev_content_len', 0)
        if prev_len > 0:
            terminal_width = shutil.get_terminal_size().columns
            prev_lines = self._get_line_count(prev_len, terminal_width)
            
            # Move up to first line if multi-line
            if prev_lines > 1:
                sys.stdout.write(f'\x1b[{prev_lines - 1}A')
            
            # Clear all lines
            for i in range(prev_lines):
                sys.stdout.write('\r\x1b[2K')
                if i < prev_lines - 1:
                    sys.stdout.write('\x1b[1B')
            
            # Go back to first line
            if prev_lines > 1:
                sys.stdout.write(f'\x1b[{prev_lines - 1}A')
            
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\x1b[2K\r')
        
        sys.stdout.flush()
        self._prev_content_len = 0
        self._prev_cursor_idx = 0

    def _custom_input(self, prompt_text):
        """
        Reads a line of user input, allowing ESC to toggle modes.
        Supports bracketed paste mode for handling Ctrl+V.
        Returns:
            tuple[str, bool]: (input_text, should_switch_mode)
        """
        # Reset content length tracking for new input (use visible length)
        self._prev_content_len = self._visible_len(prompt_text)
        self._prev_cursor_idx = 0
        
        sys.stdout.write(prompt_text)
        sys.stdout.flush()

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        tty.setraw(fd)
        
        # Enable bracketed paste mode
        sys.stdout.write('\x1b[?2004h')
        sys.stdout.flush()

        buffer = []
        cursor_idx = 0
        
        current_history = self.history[self.mode]
        history_pos = len(current_history)
        
        # Track prompt state for live resize
        self._in_input = True
        self._update_prompt_state(prompt_text, buffer, cursor_idx)

        try:
            while True:
                char = self._getch()

                # ESC toggles between COMMAND/INSTRUCT modes or starts an escape sequence
                if ord(char) == 27:
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        next_char = self._getch()
                        
                        if next_char == '[':
                            code = self._getch()
                            
                            # Check for bracketed paste start: ESC [ 2 0 0 ~
                            if code == '2':
                                # Read rest of sequence
                                rest = code
                                for _ in range(3):  # Read 00~
                                    if select.select([sys.stdin], [], [], 0.05)[0]:
                                        rest += self._getch()
                                
                                if rest == '200~':
                                    # Bracketed paste started, read until end
                                    pasted = self._read_bracketed_paste()
                                    for c in pasted:
                                        buffer.insert(cursor_idx, c)
                                        cursor_idx += 1
                                    self._redraw_line(prompt_text, buffer, cursor_idx)
                                    continue
                            
                            # Mouse events: ESC [ M ... (legacy) or ESC [ < ... (SGR)
                            if code == 'M':
                                # Legacy mouse: 3 more bytes (button+32, x+32, y+32)
                                button_byte = x_byte = y_byte = None
                                if select.select([sys.stdin], [], [], 0.05)[0]:
                                    button_byte = self._getch()
                                if select.select([sys.stdin], [], [], 0.05)[0]:
                                    x_byte = self._getch()
                                if select.select([sys.stdin], [], [], 0.05)[0]:
                                    y_byte = self._getch()
                                
                                if button_byte:
                                    button = ord(button_byte) - 32
                                    # Button 64 = scroll up, 65 = scroll down
                                    if button == 64 and hasattr(self, '_screen_buffer'):
                                        self._screen_buffer.scroll_up(1)
                                        self._screen_buffer.redraw(self._get_prompt_state())
                                    elif button == 65 and hasattr(self, '_screen_buffer'):
                                        self._screen_buffer.scroll_down(1)
                                        self._screen_buffer.redraw(self._get_prompt_state())
                                continue
                            
                            if code == '<':
                                # SGR mouse: read until M or m, format: button;x;yM
                                seq = ''
                                while select.select([sys.stdin], [], [], 0.05)[0]:
                                    c = self._getch()
                                    seq += c
                                    if c in ('M', 'm'):
                                        break
                                
                                # Parse: "button;x;yM" or "button;x;ym"
                                try:
                                    parts = seq.rstrip('Mm').split(';')
                                    if len(parts) >= 1:
                                        button = int(parts[0])
                                        # Button 64 = scroll up, 65 = scroll down
                                        if button == 64 and hasattr(self, '_screen_buffer'):
                                            self._screen_buffer.scroll_up(1)
                                            self._screen_buffer.redraw(self._get_prompt_state())
                                        elif button == 65 and hasattr(self, '_screen_buffer'):
                                            self._screen_buffer.scroll_down(1)
                                            self._screen_buffer.redraw(self._get_prompt_state())
                                except (ValueError, IndexError):
                                    pass
                                continue
                            
                            # Regular arrow key or other escape sequence
                            cursor_idx, history_pos, buffer = self._handle_arrow(
                                code, buffer, cursor_idx, current_history, history_pos, prompt_text
                            )
                            continue
                        
                        elif next_char == 'O':
                            # Application cursor mode: ESC O A/B/C/D
                            code = self._getch()
                            cursor_idx, history_pos, buffer = self._handle_arrow(
                                code, buffer, cursor_idx, current_history, history_pos, prompt_text
                            )
                            continue
                        
                        else:
                            # Unknown escape sequence - consume any remaining chars
                            while select.select([sys.stdin], [], [], 0.05)[0]:
                                self._getch()
                            continue
                    
                    # Plain ESC (no following chars) - switch modes
                    self._clear_current_input()
                    return "", True

                # Ctrl+C raises to exit immediately
                if ord(char) == 3:
                    raise KeyboardInterrupt

                # Enter commits the buffered input
                if ord(char) in [13, 10]:
                    # Move to end of content if multi-line
                    prompt_visible_len = self._visible_len(prompt_text)
                    full_len = prompt_visible_len + len(buffer)
                    terminal_width = shutil.get_terminal_size().columns
                    if terminal_width > 0:
                        total_lines = self._get_line_count(full_len, terminal_width)
                        cursor_line = self._get_line_count(prompt_visible_len + cursor_idx, terminal_width) if cursor_idx > 0 or prompt_visible_len > 0 else 1
                        lines_down = total_lines - cursor_line
                        if lines_down > 0:
                            sys.stdout.write(f'\x1b[{lines_down}B')
                    sys.stdout.write('\r\n')
                    result = "".join(buffer)
                    if result.strip():
                        self.history[self.mode].append(result)
                    self._prev_content_len = 0
                    self._prev_cursor_idx = 0
                    return result, False

                # Backspace removes the last character
                if ord(char) in [127, 8]:
                    if cursor_idx > 0:
                        buffer.pop(cursor_idx - 1)
                        cursor_idx -= 1
                        self._redraw_line(prompt_text, buffer, cursor_idx)
                    continue

                buffer.insert(cursor_idx, char)
                cursor_idx += 1
                self._redraw_line(prompt_text, buffer, cursor_idx)
        finally:
            # Mark input as complete
            self._in_input = False
            
            # Disable bracketed paste mode
            sys.stdout.write('\x1b[?2004l')
            sys.stdout.flush()
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _handle_arrow(self, code, buffer, cursor_idx, history=None, history_pos=None, prompt_text=None):
        """
        Apply visual feedback for known arrow key codes.
        Returns updated (cursor_idx, history_pos, buffer)
        """
        # Handle Up Arrow (History Back)
        if code == 'A' and history is not None and history_pos is not None:
            if history_pos > 0:
                history_pos -= 1
                buffer = list(history[history_pos])
                cursor_idx = len(buffer)
                self._redraw_line(prompt_text, buffer, cursor_idx)
            return cursor_idx, history_pos, buffer
            
        # Handle Down Arrow (History Forward)
        elif code == 'B' and history is not None and history_pos is not None:
            if history_pos < len(history):
                history_pos += 1
                if history_pos == len(history):
                    buffer = []
                else:
                    buffer = list(history[history_pos])
                cursor_idx = len(buffer)
                self._redraw_line(prompt_text, buffer, cursor_idx)
            return cursor_idx, history_pos, buffer
            
        # Handle Right Arrow
        elif code == 'C' and buffer and cursor_idx < len(buffer):
            terminal_width = shutil.get_terminal_size().columns
            if terminal_width > 0:
                # Calculate current position (use visible length, excludes \r)
                prompt_visible_len = self._visible_len(prompt_text)
                current_pos = prompt_visible_len + cursor_idx
                
                # Check if we're moving to a new line
                current_col = current_pos % terminal_width
                
                if current_col == terminal_width - 1:
                    # At end of terminal line, need to move down and to start of next line
                    sys.stdout.write('\x1b[1B')  # Move down
                    sys.stdout.write('\r')       # Go to start of line
                else:
                    # Normal move right
                    sys.stdout.write('\x1b[C')
            else:
                sys.stdout.write('\x1b[C')
            sys.stdout.flush()
            self._prev_cursor_idx = cursor_idx + 1
            return cursor_idx + 1, history_pos, buffer
            
        # Handle Left Arrow
        elif code == 'D' and buffer and cursor_idx > 0:
            terminal_width = shutil.get_terminal_size().columns
            if terminal_width > 0:
                # Calculate current position (use visible length, excludes \r)
                prompt_visible_len = self._visible_len(prompt_text)
                current_pos = prompt_visible_len + cursor_idx
                
                # Check if we're at the start of a terminal line
                current_col = current_pos % terminal_width
                
                if current_col == 0:
                    # At start of terminal line, need to move up and to end of previous line
                    sys.stdout.write('\x1b[1A')  # Move up
                    sys.stdout.write(f'\x1b[{terminal_width}G')  # Go to last column
                else:
                    # Normal move left
                    sys.stdout.write('\x1b[D')
            else:
                sys.stdout.write('\x1b[D')
            sys.stdout.flush()
            self._prev_cursor_idx = cursor_idx - 1
            return cursor_idx - 1, history_pos, buffer
            
        else:
            return cursor_idx, history_pos, buffer

    def _enter_alternate_screen(self):
        """Enter alternate screen buffer (like vim/less)."""
        sys.stdout.write('\x1b[?1049h')  # Enter alternate screen
        sys.stdout.write('\x1b[2J')       # Clear screen
        sys.stdout.write('\x1b[H')        # Move cursor to top-left
        # Enable mouse tracking for scroll wheel
        sys.stdout.write('\x1b[?1000h')   # Enable mouse button tracking
        sys.stdout.write('\x1b[?1006h')   # Enable SGR extended mouse mode
        sys.stdout.flush()
        self._in_alternate_screen = True
        
        # Activate screen buffer for content tracking
        self._screen_buffer = ScreenBuffer.get()
        self._screen_buffer.activate()
        
        # Enable resize signal handler
        live_display = LiveInspectDisplay.get()
        live_display.enable_signal()
    
    def _exit_alternate_screen(self):
        """Exit alternate screen buffer and restore normal terminal."""
        if getattr(self, '_in_alternate_screen', False):
            # Deactivate screen buffer
            if hasattr(self, '_screen_buffer'):
                self._screen_buffer.deactivate()
            
            # Disable mouse tracking
            sys.stdout.write('\x1b[?1006l')   # Disable SGR extended mouse mode
            sys.stdout.write('\x1b[?1000l')   # Disable mouse button tracking
            sys.stdout.write('\x1b[?1049l')   # Exit alternate screen
            sys.stdout.flush()
            self._in_alternate_screen = False
    
    def _buffered_print(self, text: str):
        """Print text and add to screen buffer for resize redraw."""
        print(text, end='\n\r')
        if hasattr(self, '_screen_buffer') and self._screen_buffer._active:
            self._screen_buffer.add_line(text)

    def run(self, use_alternate_screen=True):
        """
        Run the REPL.
        
        Args:
            use_alternate_screen: If True, use alternate screen buffer (like vim).
                                  This gives full control over the display but
                                  clears when exiting.
        """
        self._in_alternate_screen = False
        
        try:
            if use_alternate_screen:
                self._enter_alternate_screen()
            
            self._buffered_print(f"Starting in {self.mode} mode. Press ESC to switch modes.")
            self._buffered_print(f"Type 'help' in COMMAND mode for available commands.\n")

            while True:
                try:
                    prompt_label = ">>> " if self.mode == "COMMAND" else "> "
                    prompt_label = "\r" + prompt_label
                    user_input, switch_mode = self._custom_input(prompt_label)

                    if switch_mode:
                        self.mode = "INSTRUCT" if self.mode == "COMMAND" else "COMMAND"
                        continue

                    prompt = user_input.strip()

                    if self.mode == "COMMAND" and (prompt.lower() == 'q' or prompt.lower() == 'quit' or prompt.lower() == 'exit' or prompt.lower() == 'quit()'):
                        self._buffered_print("Exiting...")
                        break

                    if not prompt:
                        continue
                    
                    # Add the user input to buffer
                    if hasattr(self, '_screen_buffer') and self._screen_buffer._active:
                        self._screen_buffer.add_line(prompt_label.replace('\r', '') + user_input)

                    if self.mode == "COMMAND":
                        try:
                            output = self.env.execute(prompt)
                            # Buffer and print each line of output
                            if output:
                                for line in output.rstrip('\n').split('\n'):
                                    if not self.env._is_silent_result:
                                        self._buffered_print(line)
                                    else:
                                        print(line, end='\n\r')
                        except Exception as exc:
                            self._buffered_print(f"Error: {exc}")
                    else:
                        # Store the prompt and its inspection result
                        stored_prompt = self.env.add_prompt(prompt)
                        stored_prompt.result = self.inspector.inspect(prompt)

                except KeyboardInterrupt:
                    self._buffered_print("\nExiting...")
                    break
                except Exception as exc:
                    self._buffered_print(f"[!] Error processing prompt: {exc}")

        finally:
            # Always exit alternate screen on cleanup
            self._exit_alternate_screen()
            
            # Clear stdout cache and flush stdin to prevent weird characters in bash output
            sys.stdout.flush()
            try:
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            except (termios.error, AttributeError):
                pass

