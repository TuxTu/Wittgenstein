"""
REPL orchestration: mode dispatch, screen buffer integration, mode handlers.
"""
import sys
import termios

from .inspector import PromptInspector, LiveInspectDisplay
from .screen_buffer import ScreenBuffer
from .line_editor import LineEditor
from . import terminal as term
from env import ExecutionEnvironment


_PROMPT_LABELS = {"COMMAND": ">>> ", "INSTRUCT": "> ", "CHAT": "chat> "}
_MODE_CYCLE = {"COMMAND": "INSTRUCT", "INSTRUCT": "CHAT", "CHAT": "COMMAND"}
_QUIT_COMMANDS = {"q", "quit", "exit", "quit()"}


class InputProcessor:
    """
    Orchestrates the REPL experience: mode cycling, input dispatch,
    screen buffer management.
    """

    def __init__(self, model_id, default_mode="COMMAND"):
        try:
            self.inspector = PromptInspector(model_id)
        except Exception as exc:
            print(f"\r[!] Failed to initialize prompt inspector: {exc}")
            sys.exit(1)

        self.env = ExecutionEnvironment(self.inspector, model_id)
        self.mode = default_mode

        # Per-mode history buffers
        self.history = {
            "COMMAND": [],
            "INSTRUCT": [],
            "CHAT": [],
        }

        # Screen buffer (set up when entering alternate screen)
        self._screen_buffer = None
        self._in_alternate_screen = False

        # Line editor (created per input call with current history)
        self._editor = None

        # Live display resize support
        self._live_display = LiveInspectDisplay.get()

    # ------------------------------------------------------------------
    # Screen buffer helpers
    # ------------------------------------------------------------------

    def _buffered_print(self, text: str):
        """Print text and add to screen buffer for resize redraw."""
        print(text, end='\n\r')
        if self._screen_buffer and self._screen_buffer._active:
            self._screen_buffer.add_line(text)

    def _enter_alternate_screen(self):
        term.enter_alternate_screen()
        self._in_alternate_screen = True
        self._screen_buffer = ScreenBuffer.get()
        self._screen_buffer.activate()
        self._live_display.enable_signal()

    def _exit_alternate_screen(self):
        if self._in_alternate_screen:
            if self._screen_buffer:
                self._screen_buffer.deactivate()
            term.exit_alternate_screen()
            self._in_alternate_screen = False

    # ------------------------------------------------------------------
    # Scroll callback for the line editor
    # ------------------------------------------------------------------

    def _on_scroll(self, direction: int):
        """Called by LineEditor on mouse scroll. direction: 1=up, -1=down."""
        if not self._screen_buffer:
            return
        if direction > 0:
            self._screen_buffer.scroll_up(1)
        else:
            self._screen_buffer.scroll_down(1)
        self._screen_buffer.redraw(None)

    # ------------------------------------------------------------------
    # Mode handlers
    # ------------------------------------------------------------------

    def _handle_command(self, text: str):
        """Execute Python code in the persistent namespace."""
        try:
            output = self.env.execute(text)
            if output:
                for line in output.rstrip('\n').split('\n'):
                    if not self.env._is_silent_result:
                        self._buffered_print(line)
                    else:
                        print(line, end='\n\r')
        except Exception as exc:
            self._buffered_print(f"Error: {exc}")

    def _handle_instruct(self, text: str):
        """Tokenize, store, and inspect a prompt."""
        stored_prompt = self.env.add_prompt(text)
        stored_prompt.result = self.inspector.inspect(stored_prompt)

    def _handle_chat(self, text: str):
        """Send a user message and display the model's response."""
        try:
            response = self.env.chat_message(text)
            self._buffered_print(response)
        except Exception as exc:
            self._buffered_print(f"Error: {exc}")

    # ------------------------------------------------------------------
    # Main REPL loop
    # ------------------------------------------------------------------

    def run(self, use_alternate_screen=True):
        """
        Run the REPL inside the executor context.

        Args:
            use_alternate_screen: If True, use alternate screen buffer (like vim).
        """
        mode_handlers = {
            "COMMAND":  self._handle_command,
            "INSTRUCT": self._handle_instruct,
            "CHAT":     self._handle_chat,
        }

        try:
            with self.env:
                if use_alternate_screen:
                    self._enter_alternate_screen()

                self._buffered_print(
                    f"Starting in {self.mode} mode. "
                    "Press ESC to cycle modes (COMMAND → INSTRUCT → CHAT)."
                )
                self._buffered_print("Type 'help' in COMMAND mode for available commands.\n")

                while True:
                    try:
                        prompt_label = "\r" + _PROMPT_LABELS[self.mode]
                        editor = LineEditor(
                            self.history[self.mode],
                            on_scroll=self._on_scroll,
                        )
                        user_input, switch_mode = editor.read_line(prompt_label)

                        if switch_mode:
                            old_mode = self.mode
                            self.mode = _MODE_CYCLE[self.mode]
                            if old_mode == "CHAT":
                                self.env.reset_chat()
                            self._buffered_print(f"[{self.mode} mode]")
                            continue

                        text = user_input.strip()

                        if self.mode == "COMMAND" and text.lower() in _QUIT_COMMANDS:
                            self._buffered_print("Exiting...")
                            break

                        if not text:
                            continue

                        # Echo the input into the screen buffer
                        if self._screen_buffer and self._screen_buffer._active:
                            self._screen_buffer.add_line(
                                prompt_label.replace('\r', '') + user_input
                            )

                        mode_handlers[self.mode](text)

                    except KeyboardInterrupt:
                        self._buffered_print("\nExiting...")
                        break
                    except Exception as exc:
                        self._buffered_print(f"[!] Error processing prompt: {exc}")

        finally:
            self._exit_alternate_screen()
            sys.stdout.flush()
            try:
                termios.tcflush(sys.stdin, termios.TCIFLUSH)
            except (termios.error, AttributeError):
                pass
