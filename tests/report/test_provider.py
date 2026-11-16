from __future__ import annotations

from mmpbsa.report import ReportProvider, get_provider


def test_provider_contract() -> None:
    provider = get_provider()
    assert isinstance(provider, ReportProvider)
    assert provider.provider_id == "report"
    assert provider.api_version == "2"
    assert provider.package_version == "0.7.0"
    assert provider.capabilities == ("report",)
    assert provider.get_service("report") is provider
    assert provider.get_service("report.assets") is None
    assert provider.get_service("unknown") is None
