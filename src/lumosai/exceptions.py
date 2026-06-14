from __future__ import annotations


class LumosError(Exception):
    """Base exception for lumosai errors."""


class LumosValidationError(LumosError, ValueError):
    """Raised when user-provided data or arguments are invalid."""


class LumosOptionalDependencyError(LumosError, ImportError):
    """Raised when an optional dependency is required but not installed."""


class LumosConfigurationError(LumosError, RuntimeError):
    """Raised when runtime configuration prevents a requested action."""
