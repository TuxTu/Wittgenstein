"""
Low-level terminal I/O: raw mode, ANSI escape sequences, character reading.
"""
import sys
import os
import select
import shutil


def getch() -> str:
    """Read a single byte from stdin (assumes raw mode)."""
    fd = sys.stdin.fileno()
    return os.read(fd, 1).decode()


def visible_len(text: str) -> int:
    """Visible length of text (excluding control characters like \\r)."""
    return len(text.replace('\r', ''))


def line_count(text_len: int, terminal_width: int) -> int:
    """How many terminal lines a text of given length occupies."""
    if terminal_width <= 0:
        return 1
    return max(1, (text_len + terminal_width - 1) // terminal_width)


def terminal_width() -> int:
    return shutil.get_terminal_size().columns


# ---------------------------------------------------------------------------
# Escape-sequence writing helpers
# ---------------------------------------------------------------------------

def write(data: str):
    """Write to stdout without flushing."""
    sys.stdout.write(data)


def flush():
    sys.stdout.flush()


def write_flush(data: str):
    sys.stdout.write(data)
    sys.stdout.flush()


def move_up(n: int):
    if n > 0:
        write(f'\x1b[{n}A')


def move_down(n: int):
    if n > 0:
        write(f'\x1b[{n}B')


def move_to_column(col: int):
    """Move to column (1-indexed)."""
    write(f'\x1b[{col}G')


def clear_to_end():
    """Clear from cursor to end of screen."""
    write('\x1b[0J')


def clear_line():
    """Clear entire current line."""
    write('\x1b[2K')


def carriage_return():
    write('\r')


def newline():
    write('\r\n')


# ---------------------------------------------------------------------------
# Bracketed paste
# ---------------------------------------------------------------------------

def enable_bracketed_paste():
    write_flush('\x1b[?2004h')


def disable_bracketed_paste():
    write_flush('\x1b[?2004l')


def read_bracketed_paste() -> str:
    """
    Read pasted content until the bracketed paste end sequence.
    Newlines → spaces, tabs → 4 spaces.
    """
    paste_buffer = []
    while True:
        char = getch()
        if ord(char) == 27:
            seq = char
            for _ in range(5):
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    seq += getch()
                else:
                    break
            if seq == '\x1b[201~':
                break
            else:
                paste_buffer.extend(seq)
        elif ord(char) == 9:
            paste_buffer.append('    ')
        elif ord(char) in [10, 13]:
            paste_buffer.append(' ')
        else:
            paste_buffer.append(char)
    return ''.join(paste_buffer).rstrip(' ')


# ---------------------------------------------------------------------------
# Alternate screen
# ---------------------------------------------------------------------------

def enter_alternate_screen():
    write('\x1b[?1049h')  # Enter alternate screen
    write('\x1b[2J')       # Clear screen
    write('\x1b[H')        # Move cursor to top-left
    write('\x1b[?1000h')   # Enable mouse button tracking
    write('\x1b[?1006h')   # Enable SGR extended mouse mode
    flush()


def exit_alternate_screen():
    write('\x1b[?1006l')   # Disable SGR extended mouse mode
    write('\x1b[?1000l')   # Disable mouse button tracking
    write('\x1b[?1049l')   # Exit alternate screen
    flush()


# ---------------------------------------------------------------------------
# Escape sequence parsing
# ---------------------------------------------------------------------------

def has_pending_input(timeout: float = 0.1) -> bool:
    """Check if there's pending input on stdin."""
    return bool(select.select([sys.stdin], [], [], timeout)[0])


def consume_remaining():
    """Drain any remaining characters from stdin."""
    while has_pending_input(0.05):
        getch()
