from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import bash_gmx_command, mamba_command, replica_count, replica_names, run_logged, write_text_atomic


def convert_to_gromacs(paths: Any, profile: dict[str, Any]) -> None:
    paths.gromacs.mkdir(parents=True, exist_ok=True)
    run_logged(
        mamba_command(
            profile,
            [
                "acpype",
                "-p",
                str(paths.amber / "system_solvated.prmtop"),
                "-x",
                str(paths.amber / "system_solvated.inpcrd"),
                "-b",
                "system",
            ],
        ),
        paths.logs / "acpype.log",
        cwd=paths.gromacs,
    )
    acpype_dir = paths.gromacs / "system.amb2gmx"
    if not acpype_dir.exists():
        raise RuntimeError(f"ACPYPE did not create {acpype_dir}")
    align_gro_to_top_molecule_order(acpype_dir / "system_GMX.gro", acpype_dir / "system_GMX.top")
    for idx, replica in enumerate(replica_names(profile), start=1):
        rep = paths.md / replica
        rep.mkdir(parents=True, exist_ok=True)
        for source in acpype_dir.iterdir():
            if source.is_file():
                shutil.copy2(source, rep / source.name)
        for name, text in mdp_texts(profile, replica_index=idx).items():
            write_text_atomic(rep / name, text)


def align_gro_to_top_molecule_order(gro_path: Path, top_path: Path) -> None:
    """Keep ACPYPE .gro atom order consistent with GROMACS [ molecules ]."""
    molecule_order = parse_top_molecules(top_path)
    atom_counts = parse_molecule_atom_counts(top_path)
    system_name = normalize_molecule_name("system")
    if not molecule_order or system_name not in atom_counts:
        return

    gro_lines = gro_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(gro_lines) < 3:
        return
    atom_total = int(gro_lines[1].strip())
    atom_lines = gro_lines[2 : 2 + atom_total]
    box_lines = gro_lines[2 + atom_total :]
    system_atoms = atom_counts[system_name] * molecule_order.get(system_name, 0)
    if system_atoms <= 0 or system_atoms >= len(atom_lines):
        return

    prefix = atom_lines[:system_atoms]
    molecule_blocks: dict[str, list[list[str]]] = defaultdict(list)
    index = system_atoms
    while index < len(atom_lines):
        line = atom_lines[index]
        molecule_name = normalize_molecule_name(line[5:10].strip())
        atom_count = atom_counts.get(molecule_name)
        if atom_count is None or atom_count <= 0:
            return
        block = atom_lines[index : index + atom_count]
        if len(block) != atom_count:
            return
        molecule_blocks[molecule_name].append(block)
        index += atom_count

    ordered = list(prefix)
    changed = False
    for molecule_name, molecule_count in molecule_order.items():
        if molecule_name == system_name:
            continue
        blocks = molecule_blocks.get(molecule_name, [])
        if len(blocks) != molecule_count:
            return
        for block in blocks:
            ordered.extend(block)
    if len(ordered) != len(atom_lines):
        return
    changed = ordered != atom_lines
    if not changed:
        return

    renumbered = [renumber_gro_atom(line, serial) for serial, line in enumerate(ordered, start=1)]
    write_text_atomic(gro_path, "\n".join([gro_lines[0], gro_lines[1], *renumbered, *box_lines]) + "\n")


def parse_top_molecules(top_path: Path) -> dict[str, int]:
    molecules: dict[str, int] = {}
    in_molecules = False
    for line in top_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_molecules = stripped == "[ molecules ]"
            continue
        if not in_molecules or not stripped or stripped.startswith(";"):
            continue
        parts = stripped.split()
        if len(parts) >= 2 and parts[1].isdigit():
            molecules[normalize_molecule_name(parts[0])] = int(parts[1])
    return molecules


def parse_molecule_atom_counts(top_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    current_molecule: str | None = None
    section = ""
    atom_count = 0
    expect_molecule_name = False
    for line in top_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if current_molecule and section == "atoms":
                counts[current_molecule] = atom_count
            section = stripped.strip("[] ").strip()
            atom_count = 0
            expect_molecule_name = section == "moleculetype"
            continue
        if not stripped or stripped.startswith(";"):
            continue
        if expect_molecule_name:
            current_molecule = normalize_molecule_name(stripped.split()[0])
            expect_molecule_name = False
            continue
        if current_molecule and section == "atoms":
            parts = stripped.split()
            if parts and parts[0].isdigit():
                atom_count += 1
    if current_molecule and section == "atoms":
        counts[current_molecule] = atom_count
    return counts


def normalize_molecule_name(name: str) -> str:
    return name.strip().upper()


def renumber_gro_atom(line: str, serial: int) -> str:
    if len(line) < 20:
        return line
    return f"{line[:15]}{serial % 100000:5d}{line[20:]}"


def run_em(paths: Any, profile: dict[str, Any]) -> None:
    run_gmx(paths, profile, "grompp_em", ["grompp", "-f", "em.mdp", "-c", "system_GMX.gro", "-p", "system_GMX.top", "-o", "em.tpr", "-maxwarn", "2"])
    run_gmx(paths, profile, "mdrun_em", ["mdrun", "-deffnm", "em", "-ntomp", str(profile["md"]["ntomp"])])
    prefix = f"{paths.rep.name}_" if replica_count(profile) > 1 else ""
    validate_em_log(paths.logs / f"{prefix}mdrun_em.log")


def run_nvt(paths: Any, profile: dict[str, Any]) -> None:
    run_gmx(paths, profile, "grompp_nvt", ["grompp", "-f", "nvt.mdp", "-c", "em.gro", "-r", "em.gro", "-p", "system_GMX.top", "-o", "nvt.tpr", "-maxwarn", "2"])
    run_gmx(paths, profile, "mdrun_nvt", ["mdrun", "-deffnm", "nvt", "-ntomp", str(profile["md"]["ntomp"])])


def run_npt(paths: Any, profile: dict[str, Any]) -> None:
    run_gmx(
        paths,
        profile,
        "grompp_npt",
        ["grompp", "-f", "npt.mdp", "-c", "nvt.gro", "-r", "nvt.gro", "-t", "nvt.cpt", "-p", "system_GMX.top", "-o", "npt.tpr", "-maxwarn", "2"],
    )
    run_gmx(paths, profile, "mdrun_npt", ["mdrun", "-deffnm", "npt", "-ntomp", str(profile["md"]["ntomp"])])


def run_production(paths: Any, profile: dict[str, Any]) -> None:
    run_gmx(
        paths,
        profile,
        "grompp_md_prod",
        ["grompp", "-f", "md_prod.mdp", "-c", "npt.gro", "-t", "npt.cpt", "-p", "system_GMX.top", "-o", "md_prod.tpr", "-maxwarn", "2"],
    )
    run_gmx(paths, profile, "mdrun_md_prod", ["mdrun", "-deffnm", "md_prod", "-ntomp", str(profile["md"]["ntomp"])])


def run_gmx(paths: Any, profile: dict[str, Any], log_stem: str, args: list[str]) -> None:
    command = bash_gmx_command(profile, paths.rep, profile["runtime"]["gpu_id"], args)
    prefix = f"{paths.rep.name}_" if replica_count(profile) > 1 else ""
    run_logged(command, paths.logs / f"{prefix}{log_stem}.log")


def mdp_texts(profile: dict[str, Any], replica_index: int = 1) -> dict[str, str]:
    dt_ps = 0.002
    protocol = profile["protocol"]
    md = profile["md"]
    emstep = float(md.get("emstep", 0.001))
    emtol = float(md.get("emtol", 1000.0))
    nsteps_prod = int(float(protocol["production_ns"]) * 1000.0 / dt_ps)
    nstxout = max(500, int(float(protocol["xtc_interval_ps"]) / dt_ps))
    temperature = float(profile["system"]["temperature_k"])
    common = """dt                      = 0.002
cutoff-scheme           = Verlet
nstlist                 = 20
rlist                   = 1.0
coulombtype             = PME
rcoulomb                = 1.0
vdwtype                 = Cut-off
rvdw                    = 1.0
pbc                     = xyz
constraints             = h-bonds
constraint-algorithm    = lincs
DispCorr                = EnerPres
"""
    return {
        "em.mdp": f"""integrator              = steep
emtol                   = {emtol:.1f}
emstep                  = {emstep:.4f}
nsteps                  = {int(md["em_steps"])}
cutoff-scheme           = Verlet
nstlist                 = 20
rlist                   = 1.0
coulombtype             = PME
rcoulomb                = 1.0
vdwtype                 = Cut-off
rvdw                    = 1.0
pbc                     = xyz
constraints             = none
""",
        "nvt.mdp": common
        + f"""integrator              = md
nsteps                  = {int(md["nvt_steps"])}
continuation            = no
gen-vel                 = yes
gen-temp                = {temperature:.1f}
gen-seed                = {int(md["seed_base"]) + int(replica_index)}
tcoupl                  = V-rescale
tc-grps                 = System
tau-t                   = 1.0
ref-t                   = {temperature:.1f}
pcoupl                  = no
nstxout-compressed      = 5000
nstenergy               = 1000
nstlog                  = 1000
""",
        "npt.mdp": common
        + f"""integrator              = md
nsteps                  = {int(md["npt_steps"])}
continuation            = yes
gen-vel                 = no
tcoupl                  = V-rescale
tc-grps                 = System
tau-t                   = 1.0
ref-t                   = {temperature:.1f}
pcoupl                  = C-rescale
pcoupltype              = isotropic
tau-p                   = 5.0
ref-p                   = 1.0
compressibility         = 4.5e-5
nstxout-compressed      = 5000
nstenergy               = 1000
nstlog                  = 1000
""",
        "md_prod.mdp": common
        + f"""integrator              = md
nsteps                  = {nsteps_prod}
continuation            = yes
gen-vel                 = no
tcoupl                  = V-rescale
tc-grps                 = System
tau-t                   = 1.0
ref-t                   = {temperature:.1f}
pcoupl                  = C-rescale
pcoupltype              = isotropic
tau-p                   = 5.0
ref-p                   = 1.0
compressibility         = 4.5e-5
nstxout-compressed      = {nstxout}
nstenergy               = 1000
nstlog                  = 1000
""",
    }


def validate_em_log(log: Any) -> None:
    text = log.read_text(encoding="utf-8", errors="replace")
    bad_markers = [
        "force on at least one atom is not finite",
        "Maximum force     =            inf",
    ]
    if any(marker in text for marker in bad_markers):
        raise RuntimeError(f"Energy minimization did not produce a stable structure; see {log}")
