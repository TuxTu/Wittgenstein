"""
Single-line editor with history, cursor movement, and bracketed paste.

Operates in raw terminal mode. Returns the entered text and whether
ESC was pressed (mode switch).
"""
import sys
import termios
import tty
from typing import List, Tuple, Optional, Callable

from . import terminal as term


class LineEditor:
    """
    Reads one line of user input with cursor movement, history, and paste support.

    The caller provides a scroll callback for mouse-wheel events (if the REPL
    uses a screen buffer).
    """

    def __init__(
        self,
        history: List[str],
        on_scroll: Optional[Callable[[int], None]] = None,
    ):
        self._history = history
        self._on_scroll = on_scroll

        # Redraw state (persists across calls for multi-line clearing)
        self._prev_content_len = 0
        self._prev_cursor_idx = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_line(self, prompt_text: str) -> Tuple[str, bool]:
        """
        Read a line of user input.

        Returns:
            ``(text, should_switch_mode)`` — the entered text, and whether
            the user pressed plain ESC.
        """
        self._prev_content_len = term.visible_len(prompt_text)
        self._prev_cursor_idx = 0

        term.write_flush(prompt_text)

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)
        term.enable_bracketed_paste()

        buf: List[str] = []
        cursor = 0
        hist_pos = len(self._history)

        try:
            while True:
                ch = term.getch()
                o = ord(ch)

                # -- ESC: mode switch or escape sequence --
                if o == 27:
                    result = self._handle_escape(
                        prompt_text, buf, cursor, hist_pos,
                    )
                    if result is not None:
                        # Plain ESC → switch mode
                        return result
                    continue

                # -- Ctrl+C --
                if o == 3:
                    raise KeyboardInterrupt

                # -- Enter --
                if o in (13, 10):
                    self._move_cursor_to_end(prompt_text, buf, cursor)
                    term.newline()
                    text = "".join(buf)
                    if text.strip():
                        self._history.append(text)
                    self._prev_content_len = 0
                    self._prev_cursor_idx = 0
                    return text, False

                # -- Backspace --
                if o in (127, 8):
                    if cursor > 0:
                        buf.pop(cursor - 1)
                        cursor -= 1
                        self._redraw(prompt_text, buf, cursor)
                    continue

                # -- Tab → 4 spaces --
                if o == 9:
                    buf.insert(cursor, '    ')
                    cursor += 4
                    self._redraw(prompt_text, buf, cursor)
                    continue

                # -- Printable character --
                buf.insert(cursor, ch)
                cursor += 1
                self._redraw(prompt_text, buf, cursor)

        finally:
            term.disable_bracketed_paste()
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # ------------------------------------------------------------------
    # Escape-sequence dispatcher
    # ------------------------------------------------------------------

    def _handle_escape(
        self, prompt_text, buf, cursor, hist_pos,
    ) -> Optional[Tuple[str, bool]]:
        """
        Handle an ESC byte.  Returns ``("", True)`` for plain ESC (mode switch),
        or ``None`` if it was a consumed escape sequence (caller should continue).

        Mutates ``buf``, ``cursor``, ``hist_pos`` in place via the mutable list
        trick — but since Python ints are immutable we return updated values through
        the LineEditor's internal state (self._last_*).
        """
        if not term.has_pending_input():
            # Plain ESC
            self._clear_display()
            return "", True

        next_ch = term.getch()

        if next_ch == '[':
            self._handle_csi(prompt_text, buf, cursor, hist_pos)
        elif next_ch == 'O':
            code = term.getch()
            self._handle_arrow(code, prompt_text, buf, cursor, hist_pos)
        else:
            term.consume_remaining()

        return None

    def _handle_csi(self, prompt_text, buf, cursor, hist_pos):
        """Handle CSI (ESC [) sequences: arrows, paste, mouse."""
        code = term.getch()

        # Bracketed paste: ESC [ 2 0 0 ~
        if code == '2':
            rest = code
            for _ in range(3):
                if term.has_pending_input(0.05):
                    rest += term.getch()
            if rest == '200~':
                pasted = term.read_bracketed_paste()
                for c in pasted:
                    buf.insert(self._cur, c)
                    self._cur += 1
                self._redraw(prompt_text, buf, self._cur)
                return

        # Legacy mouse: ESC [ M ...
        if code == 'M':
            self._handle_legacy_mouse()
            return

        # SGR mouse: ESC [ < ...
        if code == '<':
            self._handle_sgr_mouse()
            return

        # Arrow keys / other
        self._handle_arrow(code, prompt_text, buf, cursor, hist_pos)

    # ------------------------------------------------------------------
    # Arrow keys & history
    # ------------------------------------------------------------------

    def _handle_arrow(self, code, prompt_text, buf, cursor, hist_pos):
        """Handle arrow key code. Updates self._cur / self._hist_pos / buf."""
        # Up — history back
        if code == 'A' and hist_pos > 0:
            self._hist_pos = hist_pos - 1
            buf[:] = list(self._history[self._hist_pos])
            self._cur = len(buf)
            self._redraw(prompt_text, buf, self._cur)
            return

        # Down — history forward
        if code == 'B' and hist_pos < len(self._history):
            self._hist_pos = hist_pos + 1
            if self._hist_pos == len(self._history):
                buf[:] = []
            else:
                buf[:] = list(self._history[self._hist_pos])
            self._cur = len(buf)
            self._redraw(prompt_text, buf, self._cur)
            return

        # Right
        if code == 'C' and cursor < len(buf):
            w = term.terminal_width()
            if w > 0:
                plen = term.visible_len(prompt_text)
                pos = plen + cursor
                col = pos % w
                if col == w - 1:
                    term.move_down(1)
                    term.carriage_return()
                else:
                    term.write('\x1b[C')
            else:
                term.write('\x1b[C')
            term.flush()
            self._cur = cursor + 1
            self._prev_cursor_idx = self._cur
            return

        # Left
        if code == 'D' and cursor > 0:
            w = term.terminal_width()
            if w > 0:
                plen = term.visible_len(prompt_text)
                pos = plen + cursor
                col = pos % w
                if col == 0:
                    term.move_up(1)
                    term.move_to_column(w)
                else:
                    term.write('\x1b[D')
            else:
                term.write('\x1b[D')
            term.flush()
            self._cur = cursor - 1
            self._prev_cursor_idx = self._cur
            return

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def _handle_legacy_mouse(self):
        button_byte = x_byte = y_byte = None
        if term.has_pending_input(0.05):
            button_byte = term.getch()
        if term.has_pending_input(0.05):
            x_byte = term.getch()
        if term.has_pending_input(0.05):
            y_byte = term.getch()
        if button_byte and self._on_scroll:
            button = ord(button_byte) - 32
            if button in (64, 65):
                self._on_scroll(1 if button == 64 else -1)

    def _handle_sgr_mouse(self):
        seq = ''
        while term.has_pending_input(0.05):
            c = term.getch()
            seq += c
            if c in ('M', 'm'):
                break
        try:
            parts = seq.rstrip('Mm').split(';')
            if len(parts) >= 1 and self._on_scroll:
                button = int(parts[0])
                if button in (64, 65):
                    self._on_scroll(1 if button == 64 else -1)
        except (ValueError, IndexError):
            pass

    # ------------------------------------------------------------------
    # Redraw / clear
    # ------------------------------------------------------------------

    def _redraw(self, prompt_text, buf, cursor_idx):
        """Redraw the current line and position the cursor."""
        full_line = prompt_text + "".join(buf)
        vis_len = term.visible_len(full_line)
        plen = term.visible_len(prompt_text)
        w = term.terminal_width()

        prev_cursor = self._prev_cursor_idx

        # Move to start of content
        if w > 0:
            prev_pos = plen + prev_cursor
            cur_line = prev_pos // w
        else:
            cur_line = 0

        parts = []
        if cur_line > 0:
            parts.append(f'\x1b[{cur_line}A')
        parts.append('\r')
        parts.append('\x1b[0J')
        parts.append(full_line)

        cursor_pos = plen + cursor_idx
        end_pos = plen + len(buf)

        if w > 0:
            c_line = cursor_pos // w
            c_col = cursor_pos % w
            e_line = (end_pos - 1) // w if end_pos > 0 else 0
            lines_up = e_line - c_line
            if lines_up > 0:
                parts.append(f'\x1b[{lines_up}A')
            parts.append(f'\x1b[{c_col + 1}G')
        else:
            after = len(buf) - cursor_idx
            if after > 0:
                parts.append(f'\x1b[{after}D')

        term.write_flush(''.join(parts))

        self._prev_content_len = vis_len
        self._prev_cursor_idx = cursor_idx
        # Sync cursor/hist for arrow key handler
        self._cur = cursor_idx

    def _clear_display(self):
        """Clear the current multi-line input display."""
        prev_len = self._prev_content_len
        if prev_len > 0:
            w = term.terminal_width()
            prev_lines = term.line_count(prev_len, w)
            if prev_lines > 1:
                term.write(f'\x1b[{prev_lines - 1}A')
            for i in range(prev_lines):
                term.write('\r\x1b[2K')
                if i < prev_lines - 1:
                    term.write('\x1b[1B')
            if prev_lines > 1:
                term.write(f'\x1b[{prev_lines - 1}A')
            term.write('\r')
        else:
            term.write('\x1b[2K\r')
        term.flush()
        self._prev_content_len = 0
        self._prev_cursor_idx = 0

    def _move_cursor_to_end(self, prompt_text, buf, cursor_idx):
        """Move terminal cursor to end of content before committing."""
        plen = term.visible_len(prompt_text)
        full_len = plen + len(buf)
        w = term.terminal_width()
        if w > 0:
            total_lines = term.line_count(full_len, w)
            cur_line = term.line_count(plen + cursor_idx, w) if cursor_idx > 0 or plen > 0 else 1
            lines_down = total_lines - cur_line
            if lines_down > 0:
                term.move_down(lines_down)
                term.flush()
