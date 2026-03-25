"""Project-specific exception hierarchy."""


class FrameSelectorError(Exception):
    """Base exception for all woodpecker errors."""


class LoadError(FrameSelectorError):
    """Raised when a data source cannot be loaded."""


class PipelineError(FrameSelectorError):
    """Raised when a pipeline step fails."""
