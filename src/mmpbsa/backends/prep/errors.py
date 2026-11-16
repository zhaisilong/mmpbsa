"""Errors raised before a case can be admitted as a PreparedCase."""


class PreparationError(RuntimeError):
    """A source, chemistry, or preparation invariant was violated."""

