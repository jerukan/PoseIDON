from pathlib import Path

import click

from burybarrel.foundpose_fit import load_fit_write


@click.command()
@click.option(
    "-d",
    "--datadir",
    "datadir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "-o",
    "--outdir",
    "resdir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "-m",
    "--modeldir",
    "objdir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "--coarse",
    "use_coarse",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="use coarse poses instead",
)
@click.option(
    "--icp",
    "use_icp",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="run icp before rotation averaging",
)
def run_foundpose_fit(datadir: Path, resdir: Path, objdir: Path, use_coarse: bool=False, use_icp: bool=False):
    load_fit_write(datadir, resdir, objdir, use_coarse=use_coarse, use_icp=use_icp)
