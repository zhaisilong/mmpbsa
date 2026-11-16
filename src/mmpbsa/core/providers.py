"""Versioned provider discovery without importing implementation packages."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata
from typing import Protocol, runtime_checkable

from .errors import ProviderError

PROVIDER_ENTRY_POINT_GROUP = "mmpbsa.providers.v2"
PROVIDER_API_VERSION = "2"
LOCKSTEP_SERIES = (0, 7)


@runtime_checkable
class Provider(Protocol):
    provider_id: str
    api_version: str
    package_version: str
    capabilities: Iterable[str]

    def get_service(self, capability: str) -> object | None: ...


def _series(version: str) -> tuple[int, int]:
    try:
        major, minor, *_ = version.split(".")
        return int(major), int(minor)
    except (TypeError, ValueError) as exc:
        raise ProviderError(f"invalid provider package version: {version!r}") from exc


def _validate_provider(value: object, origin: str) -> Provider:
    for attribute in (
        "provider_id",
        "api_version",
        "package_version",
        "capabilities",
        "get_service",
    ):
        if not hasattr(value, attribute):
            raise ProviderError(
                f"provider from {origin} is missing {attribute!r}"
            )
    provider = value
    provider_id = str(provider.provider_id)
    if not provider_id:
        raise ProviderError(f"provider from {origin} has an empty provider_id")
    if str(provider.api_version) != PROVIDER_API_VERSION:
        raise ProviderError(
            f"provider {provider_id!r} uses API {provider.api_version!r}; "
            f"core requires {PROVIDER_API_VERSION!r}"
        )
    if _series(str(provider.package_version)) != LOCKSTEP_SERIES:
        raise ProviderError(
            f"provider {provider_id!r} version {provider.package_version!r} "
            "is outside the core 0.7 lockstep series"
        )
    capabilities = tuple(str(item) for item in provider.capabilities)
    if not capabilities or any(not item for item in capabilities):
        raise ProviderError(
            f"provider {provider_id!r} must declare non-empty capabilities"
        )
    if len(set(capabilities)) != len(capabilities):
        raise ProviderError(
            f"provider {provider_id!r} declares duplicate capabilities"
        )
    return provider


@dataclass(frozen=True, slots=True)
class ProviderDescription:
    provider_id: str
    api_version: str
    package_version: str
    capabilities: tuple[str, ...]


class ProviderRegistry:
    """Immutable-by-default registry used by the workflow composition root."""

    def __init__(self, providers: Iterable[object] = ()) -> None:
        registered: dict[str, Provider] = {}
        for value in providers:
            provider = _validate_provider(value, "explicit registration")
            if provider.provider_id in registered:
                raise ProviderError(
                    f"duplicate provider_id: {provider.provider_id!r}"
                )
            registered[provider.provider_id] = provider
        self._providers = registered

    @property
    def provider_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def describe(self) -> tuple[ProviderDescription, ...]:
        return tuple(
            ProviderDescription(
                provider_id=provider.provider_id,
                api_version=str(provider.api_version),
                package_version=str(provider.package_version),
                capabilities=tuple(sorted(str(c) for c in provider.capabilities)),
            )
            for provider in sorted(
                self._providers.values(), key=lambda item: item.provider_id
            )
        )

    def provider(self, provider_id: str) -> Provider:
        try:
            return self._providers[provider_id]
        except KeyError as exc:
            available = ", ".join(self.provider_ids) or "(none)"
            raise ProviderError(
                f"provider {provider_id!r} is not installed; available: {available}"
            ) from exc

    def service(self, provider_id: str, capability: str) -> object:
        provider = self.provider(provider_id)
        declared = {str(item) for item in provider.capabilities}
        if capability not in declared:
            raise ProviderError(
                f"provider {provider_id!r} does not declare capability "
                f"{capability!r}"
            )
        try:
            service = provider.get_service(capability)
        except Exception as exc:
            raise ProviderError(
                f"provider {provider_id!r} failed to construct "
                f"capability {capability!r}"
            ) from exc
        if service is None:
            raise ProviderError(
                f"provider {provider_id!r} returned no service for "
                f"{capability!r}"
            )
        return service

    def configured_service(
        self,
        providers: Mapping[str, str],
        capability: str,
        *,
        required: bool = True,
    ) -> object | None:
        provider_id = providers.get(capability)
        if provider_id is None:
            if required:
                raise ProviderError(
                    f"runtime does not select a provider for {capability!r}"
                )
            return None
        return self.service(provider_id, capability)


def discover_providers() -> ProviderRegistry:
    """Load installed providers from the one supported, versioned entry point."""

    try:
        selected = metadata.entry_points(group=PROVIDER_ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - Python/importlib compatibility
        selected = metadata.entry_points().select(
            group=PROVIDER_ENTRY_POINT_GROUP
        )
    providers: list[Provider] = []
    origins: dict[str, str] = {}
    for entry_point in sorted(selected, key=lambda item: item.name):
        origin = f"entry point {entry_point.name!r}"
        try:
            value = entry_point.load()
            provider = _validate_provider(value(), origin)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"cannot load {origin}") from exc
        if provider.provider_id in origins:
            raise ProviderError(
                f"duplicate provider_id {provider.provider_id!r} from "
                f"{origins[provider.provider_id]} and {origin}"
            )
        origins[provider.provider_id] = origin
        providers.append(provider)
    return ProviderRegistry(providers)
