from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import yaml
except ModuleNotFoundError:  # Keep the CLI usable with system python.
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "configs" / "default_15ns.yaml"
DEFAULT_LIGAND_PROFILE = ROOT / "configs" / "ligand_crystal_3x5ns.yaml"
WATER_NAMES = {"HOH", "WAT", "H2O", "TIP3", "SOL"}
ENV_VAR_PATTERN = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if yaml is None:
        data = parse_simple_yaml(text, path)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise SystemExit(f"YAML root must be a mapping: {path}")
    return data


def parse_simple_yaml(text: str, path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            if not line.endswith(":"):
                raise SystemExit(f"Unsupported YAML line in {path}: {raw}")
            section = line[:-1].strip()
            current = {}
            data[section] = current
            continue
        if current is None or ":" not in line:
            raise SystemExit(f"Unsupported YAML line in {path}: {raw}")
        key, value = line.strip().split(":", 1)
        current[key.strip()] = parse_scalar(value.strip())
    return data


def parse_scalar(value: str) -> Any:
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(mark in value for mark in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def load_profile(path: Path = DEFAULT_PROFILE) -> dict[str, Any]:
    profile = load_yaml(path)
    required = {
        "protocol": ["production_ns", "xtc_interval_ps", "mmpbsa_start_ns", "mmpbsa_interval_ps", "min_mmpbsa_frames"],
        "system": ["temperature_k", "salt_molar", "solvent_padding_angstrom", "solvent_shape"],
        "amber_prep": [
            "recipe",
            "nonstandard_policy",
            "default_ligand_param_mode",
            "protein_ff",
            "ligand_ff",
            "water_ff",
            "charge_method",
            "pb_radii",
        ],
        "md": ["em_steps", "nvt_steps", "npt_steps", "seed_base", "ntomp"],
        "mmpbsa": ["mpi", "np", "keep_tmp", "internal_limit_kcal_mol", "internal_std_limit_kcal_mol"],
        "qc": [
            "receptor_rmsd_fail_angstrom",
            "ligand_rmsd_warn_angstrom",
            "peptide_rmsd_warn_angstrom",
            "native_contacts_fail_min",
            "interface_distance_fail_angstrom",
        ],
        "runtime": ["mamba_env", "gmxrc", "gmx_bin", "gpu_id"],
    }
    missing: list[str] = []
    for section, keys in required.items():
        if section not in profile or not isinstance(profile[section], dict):
            missing.append(section)
            continue
        for key in keys:
            if key not in profile[section]:
                missing.append(f"{section}.{key}")
    if missing:
        raise SystemExit(f"Missing profile keys in {path}: {', '.join(missing)}")
    return profile


def expand_runtime_env(value: Any, label: str) -> str:
    expanded = os.path.expanduser(os.path.expandvars(str(value))).strip()
    if not expanded:
        raise SystemExit(f"{label} is empty; set it in the protocol or export the matching environment variable")
    unresolved = ENV_VAR_PATTERN.search(expanded)
    if unresolved:
        variable = unresolved.group(1) or unresolved.group(2)
        raise SystemExit(f"{label} references ${variable}, but environment variable {variable} is not set")
    return expanded


def gmx_runtime(profile: dict[str, Any]) -> tuple[str, str]:
    runtime = profile["runtime"]
    return (
        expand_runtime_env(runtime["gmxrc"], "runtime.gmxrc"),
        expand_runtime_env(runtime["gmx_bin"], "runtime.gmx_bin"),
    )


def load_dataset(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Dataset is empty: {path}")
    return rows


def dataset_row(path: Path, requested_job_id: str) -> dict[str, str]:
    for row in load_dataset(path):
        candidates = {job_id(row), job_name(row)}
        for legacy_key in ("case_id", "benchmark_id", "pdb_id", "model_id"):
            if row.get(legacy_key):
                candidates.add(str(row[legacy_key]))
        if requested_job_id in candidates:
            return row
    raise SystemExit(f"{requested_job_id} not found in {path}")


def job_id(row: dict[str, str]) -> str:
    value = row.get("job_id") or row.get("case_id") or row.get("benchmark_id") or row.get("name") or row.get("pdb_id")
    if not value:
        raise SystemExit("Dataset row must define job_id, case_id, benchmark_id, name, or pdb_id.")
    value = value.strip()
    if value in {"", ".", ".."} or "/" in value:
        raise SystemExit(f"Invalid job_id for filesystem path: {value!r}")
    return value


def job_name(row: dict[str, str]) -> str:
    return (row.get("name") or row.get("pdb_id") or job_id(row)).strip()


def pdb_id(row: dict[str, str]) -> str:
    return (row.get("pdb_id") or "").strip()


def model_id(row: dict[str, str]) -> str:
    return (row.get("model_id") or row.get("af3_id") or "").strip()


def case_id(row: dict[str, str]) -> str:
    return job_id(row)


def job_dir_name(row: dict[str, str]) -> str:
    return job_id(row)


def case_name(row: dict[str, str]) -> str:
    return job_dir_name(row)


def resolve_project_path(path_text: str, base: Path = ROOT) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def split_path_list(value: str | None) -> list[str]:
    if value is None:
        return []
    out: list[str] = []
    for chunk in value.replace(";", ",").split(","):
        item = chunk.strip()
        if item:
            out.append(item)
    return out


def split_chain_spec(chains: str) -> list[str]:
    if "," in chains:
        return [item.strip() for item in chains.split(",") if item.strip()]
    return [chain for chain in chains.strip() if chain.strip()]


def ligand_chain(row: dict[str, str]) -> str:
    return (row.get("ligand_chain") or row.get("ligand_chains") or "").strip()


def ligand_chains(row: dict[str, str]) -> str:
    return ligand_chain(row)


def peptide_chains(row: dict[str, str]) -> str:
    return (row.get("peptide_chains") or row.get("peptide_chain") or row.get("ligand_chains") or "").strip()


def ligand_resname(row: dict[str, str]) -> str:
    value = (row.get("ligand_resname") or row.get("resname") or "LIG").strip().upper()
    if not re.fullmatch(r"[A-Z0-9]{1,3}", value):
        raise SystemExit(f"ligand_resname must be 1-3 alphanumeric characters for Amber residue naming: {value!r}")
    return value


def residue_id(line: str) -> tuple[str, str, str, str]:
    return (line[21].strip(), line[17:20].strip(), line[22:26].strip(), line[26].strip())


def write_protein_only_pdb(
    source: Path,
    target: Path,
    receptor_chains: str,
    ligand_chains: str,
    accepted_hetero_resnames: set[str] | None = None,
) -> list[dict[str, str]]:
    chain_order = split_chain_spec(receptor_chains) + split_chain_spec(ligand_chains)
    lines_by_chain = {chain: [] for chain in chain_order}
    dropped: dict[tuple[str, str, str, str], str] = {}
    accepted = {name.upper() for name in (accepted_hetero_resnames or set())}
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("ATOM  "):
            chain = line[21].strip()
            if chain in lines_by_chain:
                lines_by_chain[chain].append(line.rstrip() + "\n")
        elif line.startswith("HETATM"):
            chain = line[21].strip()
            if chain in lines_by_chain:
                resname = line[17:20].strip().upper()
                if resname in accepted:
                    lines_by_chain[chain].append("ATOM  " + line[6:].rstrip() + "\n")
                else:
                    dropped[residue_id(line)] = line[76:78].strip()

    output: list[str] = []
    for chain in chain_order:
        output.extend(lines_by_chain[chain])
        if lines_by_chain[chain]:
            output.append("TER\n")
    output.append("END\n")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(output), encoding="utf-8")

    return [
        {"chain": chain, "resname": resname, "resseq": resseq, "icode": icode, "element": element}
        for (chain, resname, resseq, icode), element in sorted(
            dropped.items(), key=lambda item: (item[0][0], item[0][2], item[0][1], item[0][3])
        )
    ]


def write_receptor_only_pdb(source: Path, target: Path, receptor_chains: str) -> list[dict[str, str]]:
    return write_protein_only_pdb(source, target, receptor_chains, "")


def count_selected_residues(pdb: Path, chains: str) -> int:
    chain_set = set(split_chain_spec(chains))
    residues: set[tuple[str, str, str, str]] = set()
    for line in pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM  "):
            continue
        if line[21].strip() in chain_set:
            residues.add((line[21].strip(), line[17:20].strip(), line[22:26].strip(), line[26].strip()))
    return len(residues)


def count_waters(pdb: Path) -> int:
    waters: set[tuple[str, str, str]] = set()
    for line in pdb.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        if line[17:20].strip() in WATER_NAMES:
            waters.add((line[21].strip(), line[22:26].strip(), line[26].strip()))
    return len(waters)


def pdb_residue_index(line: str) -> int:
    resseq = line[22:26].strip()
    icode = line[26].strip()
    if icode.isdigit():
        resseq = f"{resseq}{icode}"
    return int(resseq)


def residue_atoms(pdb_path: Path) -> dict[int, list[int]]:
    atoms: dict[int, list[int]] = {}
    for line in pdb_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            atom_index = int(line[6:11])
            residue_index = pdb_residue_index(line)
        except ValueError as exc:
            raise SystemExit(f"Could not parse atom/residue indices from {pdb_path}: {line}") from exc
        atoms.setdefault(residue_index, []).append(atom_index)
    if not atoms:
        raise SystemExit(f"No atoms parsed from {pdb_path}")
    return atoms


def parse_simple_mask(mask: str) -> tuple[int, int]:
    match = re.fullmatch(r":(\d+)(?:-(\d+))?", mask.strip())
    if not match:
        raise SystemExit(f"Only simple residue masks are supported: {mask}")
    first = int(match.group(1))
    last = int(match.group(2) or first)
    if first <= 0 or first > last:
        raise SystemExit(f"Invalid residue mask: {mask}")
    return first, last


def flatten_atom_range(atoms_by_residue: dict[int, list[int]], first: int, last: int) -> list[int]:
    missing = [idx for idx in range(first, last + 1) if idx not in atoms_by_residue]
    if missing:
        raise SystemExit(f"Missing residues in dry complex PDB: {missing}")
    atoms: list[int] = []
    for residue_index in range(first, last + 1):
        atoms.extend(atoms_by_residue[residue_index])
    return atoms


def frame_settings(profile: dict[str, Any]) -> dict[str, Any]:
    protocol = profile["protocol"]
    replicas = replica_count(profile)
    xtc_interval_ps = float(protocol["xtc_interval_ps"])
    start_ps = float(protocol["mmpbsa_start_ns"]) * 1000.0
    interval_ps = float(protocol["mmpbsa_interval_ps"])
    prod_ps = float(protocol["production_ns"]) * 1000.0
    startframe = int(round(start_ps / xtc_interval_ps)) + 1
    interval = max(1, int(round(interval_ps / xtc_interval_ps)))
    total_frames = int(math.floor(prod_ps / xtc_interval_ps)) + 1
    per_replica = 0 if startframe > total_frames else 1 + (total_frames - startframe) // interval
    expected = per_replica * replicas
    return {
        "startframe": startframe,
        "interval": interval,
        "total_frames": total_frames,
        "replica_count": replicas,
        "replica_indices": replica_indices(profile),
        "replica_names": replica_names(profile),
        "frames_per_replica": per_replica,
        "expected_mmpbsa_frames": expected,
    }


def replica_count(profile: dict[str, Any]) -> int:
    return len(replica_indices(profile))


def replica_indices(profile: dict[str, Any]) -> list[int]:
    protocol = profile.get("protocol", {})
    raw = protocol.get("replica_indices")
    if raw not in (None, ""):
        if isinstance(raw, (list, tuple)):
            indices = [int(item) for item in raw]
        else:
            text = str(raw).strip().strip("[]")
            indices = [int(item.strip()) for item in re.split(r"[,\s]+", text) if item.strip()]
    else:
        count = max(1, int(protocol.get("replica_count", 1)))
        indices = list(range(1, count + 1))
    if not indices:
        raise SystemExit("protocol.replica_indices must contain at least one replica index")
    if any(index <= 0 for index in indices):
        raise SystemExit(f"protocol.replica_indices must be positive integers: {indices}")
    if len(set(indices)) != len(indices):
        raise SystemExit(f"protocol.replica_indices contains duplicates: {indices}")
    return indices


def replica_names(profile: dict[str, Any]) -> list[str]:
    return [f"rep{idx:02d}" for idx in replica_indices(profile)]


def replica_seed_map(profile: dict[str, Any]) -> dict[str, int]:
    seed_base = int(profile["md"]["seed_base"])
    return {f"rep{idx:02d}": seed_base + idx for idx in replica_indices(profile)}


def profile_with_replica_indices(profile: dict[str, Any], indices: list[int], scale_min_frames: bool = False) -> dict[str, Any]:
    if not indices:
        raise SystemExit("At least one replica index is required")
    current_count = replica_count(profile)
    copied = json.loads(json.dumps(profile))
    protocol = copied.setdefault("protocol", {})
    if scale_min_frames:
        current_min = int(protocol["min_mmpbsa_frames"])
        per_replica_min = max(1, int(round(current_min / current_count)))
        protocol["min_mmpbsa_frames"] = per_replica_min * len(indices)
    protocol["replica_indices"] = list(indices)
    protocol["replica_count"] = len(indices)
    return copied


def mmpbsa_enabled(profile: dict[str, Any]) -> bool:
    return bool(profile.get("mmpbsa", {}).get("enabled", True))


def explicit_water_count(profile: dict[str, Any]) -> int:
    return max(0, int(profile.get("mmpbsa", {}).get("explicit_water_count", 0)))


def aggregate_replica_values(values_by_replica: list[dict[str, float]]) -> dict[str, float]:
    if not values_by_replica:
        return {}
    keys = sorted(set.intersection(*(set(values) for values in values_by_replica)))
    aggregated: dict[str, float] = {}
    for key in keys:
        vals = [float(values[key]) for values in values_by_replica]
        mean = sum(vals) / len(vals)
        aggregated[key] = mean
        if len(vals) > 1:
            variance = sum((value - mean) ** 2 for value in vals) / (len(vals) - 1)
            sd = math.sqrt(variance)
            aggregated[f"{key}_replica_sd"] = sd
            aggregated[f"{key}_replica_sem"] = sd / math.sqrt(len(vals))
    return aggregated


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def write_index(path: Path, groups: dict[str, list[int]]) -> None:
    lines: list[str] = []
    for name, atoms in groups.items():
        lines.append(f"[ {name} ]")
        for idx in range(0, len(atoms), 15):
            lines.append(" ".join(str(atom) for atom in atoms[idx : idx + 15]))
        lines.append("")
    write_text_atomic(path, "\n".join(lines))


def require_files(paths: list[Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths)


def remove_paths(paths: list[Path]) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


@dataclass
class CommandResult:
    returncode: int
    command: list[str] | str
    log: Path


def run_logged(
    command: list[str] | str,
    log: Path,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdin: str | None = None,
) -> CommandResult:
    log.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log.open("w", encoding="utf-8") as handle:
        handle.write(f"# started: {utc_now()}\n")
        handle.write(f"# cwd: {cwd or Path.cwd()}\n")
        handle.write(f"# command: {command if isinstance(command, str) else ' '.join(command)}\n\n")
        handle.flush()
        process = subprocess.run(
            command,
            cwd=cwd,
            env=merged_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            shell=isinstance(command, str),
            input=stdin,
        )
        handle.write(f"\n# finished: {utc_now()}\n# returncode: {process.returncode}\n")
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with code {process.returncode}; see {log}")
    return CommandResult(process.returncode, command, log)


def bash_gmx_command(profile: dict[str, Any], rep: Path, gpu_id: str | int, args: list[str]) -> str:
    gmxrc, gmx_bin = gmx_runtime(profile)
    quoted = " ".join(shlex_quote(str(arg)) for arg in args)
    script = (
        "set -eo pipefail; "
        "set +u; "
        f"source {shlex_quote(gmxrc)}; "
        "set -u; "
        f"cd {shlex_quote(str(rep))}; "
        f"CUDA_VISIBLE_DEVICES={shlex_quote(str(gpu_id))} {shlex_quote(gmx_bin)} {quoted}"
    )
    return f"bash -lc {shlex_quote(script)}"


def shlex_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


def mamba_command(profile: dict[str, Any], args: list[str]) -> list[str]:
    return ["mamba", "run", "-n", str(profile["runtime"]["mamba_env"]), *args]


def mpi_pythonpath(profile: dict[str, Any]) -> str:
    candidates: list[Path] = []
    configured = str(profile.get("runtime", {}).get("mpi4py_path", "") or "").strip()
    if configured:
        candidates.append(resolve_project_path(configured))
    candidates.append(ROOT / ".local_mpi4py")
    candidates.append(ROOT.parent / "mmpbsa_v2" / ".local_mpi4py")
    candidates.append(ROOT.parent / "mmpbsa" / "mmpbsa_v2" / ".local_mpi4py")
    paths = [str(path) for path in candidates if path.exists()]
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        paths.append(existing)
    return os.pathsep.join(paths)


def label_atom_line(line: str, receptor_last: int) -> str:
    if not line.startswith(("ATOM  ", "HETATM", "TER")) or len(line) < 26:
        return line
    try:
        residue_index = int(line[22:26])
    except ValueError:
        return line
    chain = "A" if residue_index <= receptor_last else "B"
    return f"{line[:21]}{chain}{line[22:]}"


def label_chains(input_pdb: Path, output_pdb: Path, receptor_last: int) -> None:
    lines = [label_atom_line(line, receptor_last) for line in input_pdb.read_text(encoding="utf-8", errors="replace").splitlines()]
    write_text_atomic(output_pdb, "\n".join(lines) + "\n")


@contextmanager
def temp_workdir(parent: Path, prefix: str) -> Iterator[Path]:
    parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    try:
        yield temp
    finally:
        shutil.rmtree(temp, ignore_errors=True)
