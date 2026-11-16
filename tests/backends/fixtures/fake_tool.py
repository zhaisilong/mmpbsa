#!/usr/bin/env python3
"""One deterministic fake for command-level backend tests."""

from __future__ import annotations

import pathlib
import re
import shutil
import sys


def argument(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"missing {name}") from exc


args = sys.argv[1:]
if any(item in {"--version", "-V", "-h"} for item in args):
    print("Fake scientific tool 2024.2")
    raise SystemExit(0)

if args and args[0] == "grompp":
    pathlib.Path(argument("-o")).write_text("fake tpr\n", encoding="utf-8")
    if "-pp" in args:
        shutil.copyfile(argument("-p"), argument("-pp"))
elif args and args[0] == "mdrun":
    stem = argument("-deffnm")
    for suffix in ("log", "edr", "gro", "xtc", "cpt"):
        pathlib.Path(f"{stem}.{suffix}").write_text(
            f"fake {suffix}\n",
            encoding="utf-8",
        )
elif args and args[0] == "trjconv":
    pathlib.Path(argument("-o")).write_text(
        "fake selected trajectory\n",
        encoding="utf-8",
    )
elif "-eo" in args:
    control_text = pathlib.Path(argument("-i")).read_text(encoding="utf-8")
    model_id = "pb" if re.search(r"(?im)^\s*&pb\b", control_text) else "gb"
    if "-use-mdins" in args:
        native_mdins = (
            ("_MMPBSA_gb.mdin",) if model_id == "gb" else ("_MMPBSA_pb.mdin", "_MMPBSA_pb.mdin2")
        )
        for name in native_mdins:
            path = pathlib.Path.cwd() / name
            if not path.is_file() or not path.read_text(encoding="utf-8").strip():
                raise SystemExit(f"missing native MMPBSA input: {name}")
    section = "GENERALIZED BORN" if model_id == "gb" else "POISSON BOLTZMANN"
    score = "-10.0   1.0   0.5" if model_id == "gb" else "-8.0   1.5   0.7"
    pathlib.Path(argument("-o")).write_text(
        f"{section}:\nDifferences (Complex - Receptor - Ligand):\nDELTA TOTAL   {score}\n",
        encoding="utf-8",
    )
    delta_values = ("-9.0", "-11.0") if model_id == "gb" else ("-7.0", "-9.0")
    if model_id == "gb":
        delta_header = "Frame #,VDWAALS,EEL,EGB,ESURF,DELTA TOTAL"
        delta_rows = (
            f"1,-5.0,-10.0,-3.0,0.5,{delta_values[0]}",
            f"2,-6.0,-9.0,-2.0,0.4,{delta_values[1]}",
        )
    else:
        delta_header = "Frame #,VDWAALS,EEL,EPB,ENPOLAR,EDISPER,DELTA TOTAL"
        delta_rows = (
            f"1,-5.0,-10.0,-3.0,0.5,-0.2,{delta_values[0]}",
            f"2,-6.0,-9.0,-2.0,0.4,-0.1,{delta_values[1]}",
        )
    per_frame = (
        f"{section}:\n"
        "Complex Energy Terms\n"
        "Frame #,TOTAL\n"
        "1,-100.0\n"
        "2,-101.0\n"
        "\n"
        "Receptor Energy Terms\n"
        "Frame #,TOTAL\n"
        "1,-80.0\n"
        "2,-81.0\n"
        "\n"
        "Ligand Energy Terms\n"
        "Frame #,TOTAL\n"
        "1,-10.0\n"
        "2,-9.0\n"
        "\n"
        "DELTA Energy Terms\n"
        f"{delta_header}\n"
        f"{delta_rows[0]}\n"
        f"{delta_rows[1]}\n"
    )
    pathlib.Path(argument("-eo")).write_text(
        per_frame,
        encoding="utf-8",
    )
    if "-use-mdins" in args:
        for name in native_mdins:
            (pathlib.Path.cwd() / name).unlink()
elif "-i" in args:
    script = pathlib.Path(argument("-i"))
    text = script.read_text(encoding="utf-8")
    for raw_path in re.findall(r'trajout\s+"([^"]+)"', text):
        pathlib.Path(raw_path).write_text(
            "MODEL        1\nENDMDL\n",
            encoding="utf-8",
        )
    for raw_path in re.findall(r'writedata\s+"([^"]+)"', text):
        path = pathlib.Path(raw_path)
        if not path.is_absolute():
            path = pathlib.Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "#Frame RMSD\n1 0.5\n2 2.0\n",
            encoding="utf-8",
        )
    for raw_path in re.findall(r"\bout\s+(?:\"([^\"]+)\"|(\S+))", text):
        selected = next(item for item in raw_path if item)
        path = pathlib.Path(selected)
        if not path.is_absolute():
            path = pathlib.Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "native_contacts.dat":
            content = "#Frame primary_binding[native] primary_binding[mindist]\n1 2 3.0\n2 1 4.0\n"
        elif path.name == "interface_rmsd.dat":
            content = "#Frame interface_backbone\n1 0.5\n2 2.0\n"
        elif path.name == "ligand_rmsd.dat":
            content = "#Frame ligand_primary\n1 0.5\n2 2.0\n"
        elif path.name.endswith("_ligand_equilibrated_rmsd.dat"):
            content = "#Frame ligand_equilibrated\n1 0.5\n2 2.0\n"
        else:
            content = "#Frame RMSD\n1 0.5\n2 2.0\n"
        path.write_text(content, encoding="utf-8")
else:
    raise SystemExit("unsupported fake invocation: " + " ".join(args))
