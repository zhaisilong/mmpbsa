"""Exceptions raised by :mod:`mmpbsa.utils`."""

from __future__ import annotations


class MmpbsaUtilsError(Exception):
    """Base class for utility package errors."""


class JsonEncodingError(MmpbsaUtilsError, TypeError):
    """A value cannot be represented by the supported JSON contract."""


class UnsafePathError(MmpbsaUtilsError, ValueError):
    """A path is absolute, traverses its root, or resolves outside its root."""


class AtomicReplaceError(MmpbsaUtilsError, OSError):
    """An atomic filesystem publication could not be completed safely."""


class LockError(MmpbsaUtilsError):
    """Base class for lock acquisition and ownership failures."""


class LockTimeoutError(LockError, TimeoutError):
    """A file lock was not acquired before its timeout."""


class LockOwnershipError(LockError):
    """A lock changed owner before it could be refreshed or released."""


class CommandError(MmpbsaUtilsError, RuntimeError):
    """Base class for command execution failures."""
