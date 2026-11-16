"""Built-in 0.7 provider satisfying the versioned core provider ABI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .api import BackendFactory, Capability, CommandRunner
from .errors import ProviderError
from .services import BuiltinStageService, create_stage_service


@dataclass(frozen=True, slots=True)
class BuiltinProvider:
    _factories: Mapping[str, BackendFactory]
    _services: Mapping[str, BuiltinStageService]

    @property
    def provider_id(self) -> str:
        return "backends"

    @property
    def api_version(self) -> str:
        return "2"

    @property
    def package_version(self) -> str:
        return "0.7.0"

    @property
    def capabilities(self) -> tuple[str, ...]:
        return tuple(sorted(self._services))

    def get_service(self, capability: str) -> BuiltinStageService | None:
        return self._services.get(str(capability))

    def create(
        self,
        capability: str | Capability,
        config: Mapping[str, object] | None = None,
        *,
        runner: CommandRunner | None = None,
    ) -> object:
        key = capability.value if isinstance(capability, Capability) else str(capability)
        try:
            factory = self._factories[key]
        except KeyError as exc:
            raise ProviderError(
                f"provider {self.provider_id!r} does not support {key!r}"
            ) from exc
        return factory(dict(config or {}), runner=runner)


def create_provider(*, runner: CommandRunner | None = None) -> BuiltinProvider:
    # Imports are intentionally localized so provider discovery does not import
    # optional ParmEd or touch any executable.
    from .ambertools import create_ambertools_scoring_backend
    from .cpptraj import create_cpptraj_qc_backend
    from .display import create_display_asset_backend
    from .gromacs import create_gromacs_md_backend
    from .parmed import create_parmed_admission_backend
    from .trajectory import create_trajectory_backend

    return BuiltinProvider(
        {
            Capability.MD.value: create_gromacs_md_backend,
            Capability.TRAJECTORY.value: create_trajectory_backend,
            Capability.SCORING.value: (
                create_ambertools_scoring_backend
            ),
            Capability.TRAJECTORY_QC.value: create_cpptraj_qc_backend,
            Capability.ADMISSION.value: create_parmed_admission_backend,
            Capability.REPORT_ASSETS.value: create_display_asset_backend,
        },
        {
            capability.value: create_stage_service(
                capability.value,
                runner=runner,
            )
            for capability in Capability
        },
    )


def get_provider() -> BuiltinProvider:
    """Entry-point factory; importing it never probes tools."""

    return create_provider()
