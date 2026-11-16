from __future__ import annotations

from dataclasses import dataclass

import pytest

import mmpbsa.core as core
from mmpbsa.core.errors import ProviderError
from mmpbsa.core.providers import (
    LOCKSTEP_SERIES,
    PROVIDER_API_VERSION,
    PROVIDER_ENTRY_POINT_GROUP,
    ProviderRegistry,
)


@dataclass
class FakeProvider:
    provider_id: str = "fake"
    api_version: str = "2"
    package_version: str = "0.7.0"
    capabilities: tuple[str, ...] = ("md",)
    service_value: object | None = None

    def get_service(self, capability: str) -> object | None:
        if capability not in self.capabilities:
            return None
        return self.service_value


def test_provider_abi_accepts_only_core_api_and_0_7_lockstep() -> None:
    assert core.__version__ == "0.7.0"
    assert PROVIDER_ENTRY_POINT_GROUP == "mmpbsa.providers.v2"
    assert PROVIDER_API_VERSION == "2"
    assert LOCKSTEP_SERIES == (0, 7)

    service = object()
    registry = ProviderRegistry(
        [
            FakeProvider(
                provider_id="stable",
                package_version="0.7.99",
                capabilities=("md", "trajectory"),
                service_value=service,
            )
        ]
    )

    assert registry.provider_ids == ("stable",)
    assert registry.service("stable", "md") is service
    assert registry.describe()[0].package_version == "0.7.99"

    with pytest.raises(ProviderError, match="uses API"):
        ProviderRegistry([FakeProvider(api_version="1")])
    with pytest.raises(ProviderError, match="outside the core 0.7"):
        ProviderRegistry([FakeProvider(package_version="0.6.99")])
    with pytest.raises(ProviderError, match="outside the core 0.7"):
        ProviderRegistry([FakeProvider(package_version="1.4.0")])


def test_provider_capabilities_and_ids_are_unambiguous() -> None:
    with pytest.raises(ProviderError, match="duplicate capabilities"):
        ProviderRegistry(
            [FakeProvider(capabilities=("scoring", "scoring"))]
        )
    with pytest.raises(ProviderError, match="duplicate provider_id"):
        ProviderRegistry(
            [
                FakeProvider(service_value=object()),
                FakeProvider(service_value=object()),
            ]
        )

    registry = ProviderRegistry(
        [FakeProvider(capabilities=("md",), service_value=None)]
    )
    with pytest.raises(ProviderError, match="returned no service"):
        registry.service("fake", "md")
    with pytest.raises(ProviderError, match="does not declare capability"):
        registry.service("fake", "qc")
    with pytest.raises(ProviderError, match="runtime does not select"):
        registry.configured_service({}, "md")
