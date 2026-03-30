import click

from burybarrel.scripts.create_masks import create_masks
from burybarrel.scripts.reconstruct_colmap import reconstruct_colmap
from burybarrel.scripts.get_footage_keyframes import get_footage_keyframes, subset_footage_keyframes
from burybarrel.scripts.run_foundpose_fit import run_foundpose_fit
from burybarrel.scripts.run_foundpose import run_foundpose
from burybarrel.scripts.full_pipeline import run_full_pipelines
from burybarrel.scripts.pipeline_metrics import get_metrics
from burybarrel.scripts.gt_from_blender import gt_from_blender
from burybarrel.scripts.reconstruct_fast3r import reconstruct_fast3r
from burybarrel.scripts.reconstruct_vggt import reconstruct_vggt


@click.group()
def cli():
    pass


@cli.command()
@click.option("-i", "stuff", multiple=True)
def test(stuff):
    print(stuff)


@cli.command()
def datagen_train_run():
    from burybarrel.scripts import datagen_occ, train_barrelnet

    datagen_occ.run()
    train_barrelnet.run()


@cli.command()
def run_pointnet_inference():
    from burybarrel.scripts import run_pointnet_inf

    run_pointnet_inf.run()


cli.add_command(create_masks)
cli.add_command(get_footage_keyframes)
cli.add_command(subset_footage_keyframes)
cli.add_command(reconstruct_colmap)
cli.add_command(run_foundpose_fit)
cli.add_command(run_foundpose)
cli.add_command(run_full_pipelines)
cli.add_command(get_metrics)
cli.add_command(gt_from_blender)
cli.add_command(reconstruct_fast3r)
cli.add_command(reconstruct_vggt)


if __name__ == "__main__":
    cli()
