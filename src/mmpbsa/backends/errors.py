"""Backend-specific failures with stable, workflow-facing categories."""

from __future__ import annotations


class BackendError(RuntimeError):
    """Base class for backend failures."""


class BackendConfigurationError(BackendError, ValueError):
    """A backend or request was configured inconsistently."""


class BackendExecutionError(BackendError):
    """An external command failed."""


class BackendOutputError(BackendError):
    """A command completed but its declared outputs are invalid."""


class ToolProbeError(BackendError):
    """An executable could not provide a parseable version."""


class IncompatibleToolVersion(ToolProbeError):
    """A parsed tool major is outside the adapter's declared support set."""


class ProviderError(BackendError):
    """A provider cannot be registered or cannot create a capability."""
