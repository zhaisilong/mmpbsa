"""Executable scientific-tool providers for mmpbsa 0.7."""

from .api import (
    BackendReceipt,
    Capability,
    CommandRecord,
    CommandRunner,
    CommandSpec,
    GeneratedInput,
    ReceiptState,
)
from .errors import (
    BackendConfigurationError,
    BackendError,
    BackendExecutionError,
    BackendOutputError,
    IncompatibleToolVersion,
    ProviderError,
    ToolProbeError,
)
from .versioning import (
    MajorVersionPolicy,
    ToolConfig,
    ToolIdentity,
    parse_version,
    probe_tool,
)

__all__ = [
    "BackendConfigurationError",
    "BackendError",
    "BackendExecutionError",
    "BackendOutputError",
    "BackendReceipt",
    "BuiltinProvider",
    "Capability",
    "CommandRecord",
    "CommandRunner",
    "CommandSpec",
    "GeneratedInput",
    "IncompatibleToolVersion",
    "MajorVersionPolicy",
    "ProviderError",
    "ReceiptState",
    "ToolConfig",
    "ToolIdentity",
    "ToolProbeError",
    "get_provider",
    "create_provider",
    "parse_version",
    "probe_tool",
]

__version__ = "0.7.0"


def __getattr__(name: str) -> object:
    """Keep command helpers importable without eagerly importing core."""

    if name in {"BuiltinProvider", "create_provider", "get_provider"}:
        from .provider import BuiltinProvider, create_provider, get_provider

        return {
            "BuiltinProvider": BuiltinProvider,
            "create_provider": create_provider,
            "get_provider": get_provider,
        }[name]
    raise AttributeError(name)
