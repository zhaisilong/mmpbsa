from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Type

import click

from .aggregate import aggregate_run_dir
from .common import DEFAULT_LIGAND_PROFILE, DEFAULT_PROFILE, frame_settings, gmx_runtime, load_profile, mpi_pythonpath, profile_with_replica_indices, shlex_quote
from .ligand_pipeline import LigandPipeline
from .peptide_pipeline import PeptidePipeline
from .postprocess_sweep import parse_epsilons, peptide_postprocess_sweep
from .replica_merge import merge_ligand_replicas, merge_peptide_replicas
from .runner import DoneFileRunner, apply_env_overrides, discover_job_contexts
from .visualize import bundle_pymol, visualize_job, visualize_run


def protocol_option(function=None, *, default: Path = DEFAULT_PROFILE):
    def decorator(inner):
        return click.option(
            "--protocol",
            "protocol_path",
            type=click.Path(path_type=Path, dir_okay=False),
            default=default,
            show_default=True,
            help="YAML protocol file with MD/MMPBSA settings and runtime paths.",
        )(inner)

    if function is None:
        return decorator
    return decorator(function)


def peptide_protocol_option(function):
    return protocol_option(function, default=DEFAULT_PROFILE)


def ligand_protocol_option(function):
    return protocol_option(function, default=DEFAULT_LIGAND_PROFILE)


def run_pipeline(
    pipeline_cls: Type[DoneFileRunner],
    run_dir: Path,
    protocol_path: Path,
    job_id: str | None,
    mode: str,
    resume: bool,
    force: bool,
    replica_index: int | None = None,
) -> None:
    contexts = discover_job_contexts(run_dir, protocol_path, job_id=job_id)
    for context in contexts:
        if replica_index is not None:
            context = replace(context, protocol=profile_with_replica_indices(context.protocol, [replica_index], scale_min_frames=True))
        pipeline_cls(context).run(mode=mode, resume=resume, force=force)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """Unified protein-peptide and protein-ligand MMPBSA pipeline."""


@cli.group()
def peptide() -> None:
    """Protein-peptide workflow."""


@peptide.command("run")
@click.argument("run_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--force", is_flag=True, help="Clear selected mode and downstream done files/outputs before running.")
@click.option("--resume", is_flag=True, help="Skip steps that already have .<step>_done.")
@click.option("--mode", type=click.Choice(["full", "prepare", "md", "analysis", "report"]), default="full", show_default=True, help="Stage group to run.")
@click.option("--job-id", help="Run only RUN_DIR/<job-id>/<job-id>.json.")
@click.option("--replica-index", type=click.IntRange(1), help="Run one explicit replica index, for example 4 creates rep04 with seed_base+4.")
@peptide_protocol_option
def peptide_run(run_dir: Path, protocol_path: Path, job_id: str | None, replica_index: int | None, mode: str, resume: bool, force: bool) -> None:
    run_pipeline(PeptidePipeline, run_dir, protocol_path, job_id, mode, resume, force, replica_index=replica_index)


@peptide.command("merge-replicas")
@click.argument("output_job_dir", type=click.Path(path_type=Path, file_okay=False))
@click.argument("source_job_dirs", nargs=-1, type=click.Path(path_type=Path, file_okay=False))
@click.option("--force", is_flag=True, help="Overwrite merged audit/summary outputs if they already exist.")
def peptide_merge_replicas(output_job_dir: Path, source_job_dirs: tuple[Path, ...], force: bool) -> None:
    report = merge_peptide_replicas(output_job_dir, list(source_job_dirs), force=force)
    click.echo(json.dumps(report, indent=2))


@peptide.command("sweep-postprocess")
@click.argument("run_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True, help="Directory for sweep work files and reports.")
@click.option("--assay-dir", type=click.Path(path_type=Path, file_okay=False), help="Directory with assay_*.yaml files for correlation reporting.")
@click.option("--epsilons", default="4,8,12,20", show_default=True, help="Comma-separated internal dielectric values.")
@click.option("--salt-molar", type=float, default=0.150, show_default=True, help="GB saltcon and PB istrng value in molarity.")
@click.option("--np", "np_ranks", type=click.IntRange(1), default=16, show_default=True, help="MPI ranks for recomputed MMPBSA jobs.")
@click.option("--max-workers", type=click.IntRange(1), default=1, show_default=True, help="Number of independent replica MMPBSA jobs to run concurrently.")
@click.option("--job-id", help="Run or inspect only one job directory.")
@click.option("--force", is_flag=True, help="Remove existing recomputed work directories before running.")
@click.option("--dry-run", is_flag=True, help="Discover jobs and write a manifest without running MMPBSA.")
def peptide_sweep_postprocess(
    run_dir: Path,
    output_dir: Path,
    assay_dir: Path | None,
    epsilons: str,
    salt_molar: float,
    np_ranks: int,
    max_workers: int,
    job_id: str | None,
    force: bool,
    dry_run: bool,
) -> None:
    report = peptide_postprocess_sweep(
        run_dir,
        output_dir,
        assay_dir,
        epsilons=parse_epsilons(epsilons),
        salt_molar=salt_molar,
        np_ranks=np_ranks,
        max_workers=max_workers,
        job_id=job_id,
        force=force,
        dry_run=dry_run,
    )
    click.echo(json.dumps(report, indent=2))


@cli.group()
def ligand() -> None:
    """Protein-small-molecule workflow."""


@ligand.command("run")
@click.argument("run_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--force", is_flag=True, help="Clear selected mode and downstream done files/outputs before running.")
@click.option("--resume", is_flag=True, help="Skip steps that already have .<step>_done.")
@click.option("--mode", type=click.Choice(["full", "prepare", "md", "analysis", "report"]), default="full", show_default=True, help="Stage group to run.")
@click.option("--job-id", help="Run only RUN_DIR/<job-id>/<job-id>.json.")
@ligand_protocol_option
@click.option("--replica-index", type=click.IntRange(1), help="Run one explicit replica index, for example 4 creates rep04 with seed_base+4.")
def ligand_run(run_dir: Path, protocol_path: Path, job_id: str | None, replica_index: int | None, mode: str, resume: bool, force: bool) -> None:
    run_pipeline(LigandPipeline, run_dir, protocol_path, job_id, mode, resume, force, replica_index=replica_index)


@ligand.command("merge-replicas")
@click.argument("output_job_dir", type=click.Path(path_type=Path, file_okay=False))
@click.argument("source_job_dirs", nargs=-1, type=click.Path(path_type=Path, file_okay=False))
@click.option("--force", is_flag=True, help="Overwrite merged audit/summary outputs if they already exist.")
def ligand_merge_replicas(output_job_dir: Path, source_job_dirs: tuple[Path, ...], force: bool) -> None:
    report = merge_ligand_replicas(output_job_dir, list(source_job_dirs), force=force)
    click.echo(json.dumps(report, indent=2))


@cli.command()
@click.argument("run_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--job-id", help="Show only RUN_DIR/<job-id>.")
def status(run_dir: Path, job_id: str | None) -> None:
    root = run_dir.resolve()
    if not root.exists():
        raise click.ClickException(f"RUN_DIR does not exist: {root}")
    job_dirs = [root / job_id] if job_id else sorted(path for path in root.iterdir() if path.is_dir())
    for directory in job_dirs:
        if not directory.exists():
            raise click.ClickException(f"Job directory does not exist: {directory}")
        config_path = directory / f"{directory.name}.json"
        if not config_path.exists():
            if job_id:
                raise click.ClickException(f"Missing job config: {config_path}")
            continue
        done = sorted(path.name for path in directory.glob(".*_done"))
        failed = sorted(path.name for path in directory.glob(".*_failed"))
        result = directory / "result" / "summary.json"
        status_text = "complete" if result.exists() else "incomplete"
        click.echo(f"{directory.name}: {status_text}")
        click.echo(f"  done: {', '.join(done) if done else '-'}")
        click.echo(f"  failed: {', '.join(failed) if failed else '-'}")


@cli.command()
@click.argument("run_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True)
def aggregate(run_dir: Path, output_dir: Path) -> None:
    if not run_dir.exists():
        raise click.ClickException(f"RUN_DIR does not exist: {run_dir.resolve()}")
    report = aggregate_run_dir(run_dir, output_dir)
    click.echo(json.dumps(report, indent=2))


@cli.group()
def visualize() -> None:
    """Generate lightweight QC plots and portable PyMOL bundles."""


@visualize.command("job")
@click.argument("job_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True, help="Directory for generated HTML/SVG files.")
@click.option("--pymol", is_flag=True, help="Add PyMOL visualization assets to the report.")
@click.option("--export-visual", is_flag=True, help="Deprecated alias for --pymol.")
@click.option("--align/--no-align", default=True, show_default=True, help="Generate aligned visual PDB assets when --export-visual is used.")
@click.option("--movie-stride", type=click.IntRange(1), default=5, show_default=True, help="Frame stride for aligned trajectory/movie PDB export.")
@click.option("--movie", is_flag=True, help="Render and include movie.mp4 when local PyMOL and ffmpeg are available. Implies --pymol.")
@click.option("--render-video", is_flag=True, help="Deprecated alias for --movie.")
@click.option("--zip", "zip_archive", is_flag=True, help="Zip the complete generated report directory.")
@click.option("--archive-name", help="Zip filename or stem for the complete generated report. Implies --zip.")
def visualize_job_cmd(
    job_dir: Path,
    output_dir: Path,
    pymol: bool,
    export_visual: bool,
    align: bool,
    movie_stride: int,
    movie: bool,
    render_video: bool,
    zip_archive: bool,
    archive_name: str | None,
) -> None:
    render_movie = movie or render_video
    write_pymol = pymol or export_visual or render_movie
    report = visualize_job(
        job_dir,
        output_dir,
        export_visual=write_pymol,
        align=align,
        movie_stride=movie_stride,
        render_video=render_movie,
        zip_archive=zip_archive,
        archive_name=archive_name,
    )
    click.echo(json.dumps(report, indent=2))


@visualize.command("run")
@click.argument("run_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True, help="Directory for generated run-level plots.")
@click.option("--job-id", multiple=True, help="Restrict plotting to selected completed jobs. May be repeated.")
@click.option("--sort-by", default="composite", show_default=True, help="Ranking order: composite or a numeric result field.")
@click.option("--limit", type=click.IntRange(1), help="Plot at most this many jobs after sorting/filtering.")
@click.option("--include-samples", is_flag=True, help="Also write linked per-sample QC reports under OUTPUT_DIR/samples/.")
@click.option("--include-jobs", is_flag=True, help="Deprecated alias for --include-samples.")
@click.option("--pymol", is_flag=True, help="Add PyMOL visualization assets to each sample report. Implies --include-samples.")
@click.option("--export-pymol", is_flag=True, help="Deprecated alias for --pymol.")
@click.option("--align/--no-align", default=True, show_default=True, help="Generate aligned visual PDB assets when --export-pymol is used.")
@click.option("--movie-stride", type=click.IntRange(1), default=5, show_default=True, help="Frame stride for aligned trajectory/movie PDB export.")
@click.option("--movie", is_flag=True, help="Render and include movie.mp4 when local PyMOL and ffmpeg are available. Implies --pymol.")
@click.option("--render-video", is_flag=True, help="Deprecated alias for --movie.")
@click.option("--correlation-manifest", type=click.Path(path_type=Path, dir_okay=False), help="Optional CSV/TSV manifest with numeric fields to correlate against MMPBSA scores.")
@click.option("--correlate-column", multiple=True, help="Manifest column to correlate. May be repeated; default auto-detects numeric non-identifier columns.")
@click.option("--zip", "zip_archive", is_flag=True, help="Zip the complete generated report directory.")
@click.option("--archive-name", help="Zip filename or stem for the complete generated report. Implies --zip.")
def visualize_run_cmd(
    run_dir: Path,
    output_dir: Path,
    job_id: tuple[str, ...],
    sort_by: str,
    limit: int | None,
    include_samples: bool,
    include_jobs: bool,
    pymol: bool,
    export_pymol: bool,
    align: bool,
    movie_stride: int,
    movie: bool,
    render_video: bool,
    correlation_manifest: Path | None,
    correlate_column: tuple[str, ...],
    zip_archive: bool,
    archive_name: str | None,
) -> None:
    render_movie = movie or render_video
    write_pymol = pymol or export_pymol or render_movie
    report = visualize_run(
        run_dir,
        output_dir,
        job_ids=list(job_id) or None,
        sort_by=sort_by,
        limit=limit,
        include_samples=include_samples or include_jobs or write_pymol,
        export_pymol=write_pymol,
        align=align,
        movie_stride=movie_stride,
        render_video=render_movie,
        correlation_manifest=correlation_manifest,
        correlate_columns=list(correlate_column) or None,
        zip_archive=zip_archive,
        archive_name=archive_name,
    )
    click.echo(json.dumps(report, indent=2))


@visualize.command("bundle")
@click.argument("run_dir", type=click.Path(path_type=Path, file_okay=False))
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False), required=True, help="Directory for the portable bundle directory.")
@click.option("--job-id", multiple=True, required=True, help="Job to include in the bundle. May be repeated.")
@click.option("--zip", "zip_archive", is_flag=True, help="Also create a zip archive. By default only the portable directory is written.")
@click.option("--archive-name", help="Zip filename or stem. Implies --zip; defaults to pymol_bundle when omitted.")
@click.option("--snapshots-only", is_flag=True, help="Exclude pymol_trajectory.pdb and package first/mid/last snapshots only.")
@click.option("--align/--no-align", default=True, show_default=True, help="Generate aligned visual PDB assets using receptor-fit cpptraj alignment.")
@click.option("--movie-stride", type=click.IntRange(1), default=5, show_default=True, help="Frame stride for aligned trajectory/movie PDB export.")
@click.option("--keep-plots", is_flag=True, help="Keep duplicate per-job HTML/SVG plot files inside the bundle.")
@click.option("--render-video", is_flag=True, help="Try to render movie.mp4 when local pymol and ffmpeg are available.")
def visualize_bundle_cmd(
    run_dir: Path,
    output_dir: Path,
    job_id: tuple[str, ...],
    archive_name: str | None,
    zip_archive: bool,
    snapshots_only: bool,
    align: bool,
    movie_stride: int,
    keep_plots: bool,
    render_video: bool,
) -> None:
    report = bundle_pymol(
        run_dir,
        output_dir,
        job_ids=list(job_id),
        archive_name=archive_name,
        zip_archive=zip_archive,
        snapshots_only=snapshots_only,
        align=align,
        movie_stride=movie_stride,
        keep_plots=keep_plots,
        render_video=render_video,
    )
    click.echo(json.dumps(report, indent=2))


@cli.command("frame-settings")
@protocol_option
def frame_settings_cmd(protocol_path: Path) -> None:
    click.echo(json.dumps(frame_settings(load_profile(protocol_path)), indent=2))


@cli.command()
@protocol_option
def doctor(protocol_path: Path) -> None:
    profile = apply_env_overrides(load_profile(protocol_path))
    env = str(profile["runtime"]["mamba_env"])
    gmxrc, gmx_bin = gmx_runtime(profile)
    checks = [
        ["mamba", "run", "-n", env, "which", "MMPBSA.py"],
        ["mamba", "run", "-n", env, "which", "MMPBSA.py.MPI"],
        ["mamba", "run", "-n", env, "which", "cpptraj"],
        ["mamba", "run", "-n", env, "which", "acpype"],
        ["mamba", "run", "-n", env, "which", "antechamber"],
        ["mamba", "run", "-n", env, "which", "parmchk2"],
        ["mamba", "run", "-n", env, "which", "parmed"],
        ["bash", "-lc", f"source {shlex_quote(gmxrc)} && which {shlex_quote(gmx_bin)}"],
    ]
    if bool(profile["mmpbsa"]["mpi"]):
        checks[2:2] = [
            ["mamba", "run", "-n", env, "which", "mpirun"],
            ["mamba", "run", "-n", env, "python", "-c", "import mpi4py; print(mpi4py.__version__)"],
        ]
    for command in checks:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        status_text = "ok" if result.returncode == 0 else "missing"
        click.echo(f"{status_text:8s} {' '.join(command)}")
        if result.stdout.strip():
            click.echo(result.stdout.strip())
    click.echo(f"PYTHONPATH for MPI: {mpi_pythonpath(profile) or '(empty)'}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
