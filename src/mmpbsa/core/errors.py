"""Public exception hierarchy for the 0.7 workflow core."""

from __future__ import annotations


class MmpbsaError(RuntimeError):
    """Base error for stable, user-facing workflow failures."""


class ConfigurationError(MmpbsaError):
    """A versioned workflow/runtime document is invalid."""


class ContractError(MmpbsaError):
    """A typed scientific or artifact contract is invalid."""


class ArtifactIntegrityError(ContractError):
    """An artifact is absent, changed, unsafe, or inconsistent."""


class ProviderError(MmpbsaError):
    """A requested provider or capability is unavailable/incompatible."""


class WorkspaceError(MmpbsaError):
    """A workspace record or transition violates the execution model."""


class ActiveAttemptError(WorkspaceError):
    """Normal execution found an active attempt that requires explicit resume."""


class ResumeError(WorkspaceError):
    """An active attempt cannot be resumed safely."""


class RetryError(WorkspaceError):
    """A requested retry/new-attempt action is not valid."""


class TaskExecutionError(MmpbsaError):
    """A stage backend failed after an attempt had been reserved."""
