from __future__ import annotations

from pathlib import Path

import pytest

from mmpbsa.core.errors import ContractError
from mmpbsa.core.md_inputs import topology_include_closure


def test_gromacs_include_closure_is_recursive_and_relative(tmp_path: Path) -> None:
    files = {
        Path("md/topol.top"): b'#include "ff/forcefield.itp"\n',
        Path("md/ff/forcefield.itp"): b'#include "../ligand.itp"\n',
        Path("md/ligand.itp"): b"[ moleculetype ]\n",
    }
    sealed = {}
    for relative, data in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        sealed[relative] = path

    assert topology_include_closure(Path("md/topol.top"), sealed) == (
        Path("md/ff/forcefield.itp"),
        Path("md/ligand.itp"),
    )


@pytest.mark.parametrize(
    ("directive", "message"),
    [
        ('#include "missing.itp"', "undeclared"),
        ('#include "../../outside.itp"', "escapes package"),
        ("#include <forcefield.itp>", "quoted literal"),
        ("#include FF_FILE", "quoted literal"),
        ('#include "/etc/passwd"', "unsafe"),
    ],
)
def test_gromacs_include_closure_fails_closed(
    tmp_path: Path, directive: str, message: str
) -> None:
    topology = tmp_path / "topol.top"
    topology.write_text(directive + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match=message):
        topology_include_closure(Path("topol.top"), {Path("topol.top"): topology})
