class CodeModeError(RuntimeError):
    """Base error for Kedi's adapter-neutral CodeMode surface."""


class CodeModeCatalogError(CodeModeError):
    """The executable tool catalog is invalid or ambiguous."""


class CodeModeCursorError(CodeModeError):
    """A search cursor is invalid, stale, or belongs to another query."""


class CodeModeHydrationError(CodeModeError):
    """A tool schema could not be hydrated for sandbox execution."""


class CodeModeExecutionError(CodeModeError):
    """Sandbox code could not be executed safely."""


class CodeModeBoundaryError(CodeModeExecutionError):
    """A value cannot cross the tool-to-sandbox boundary safely."""
