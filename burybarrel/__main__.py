import click

from burybarrel.scripts import (
    create_masks, reconstruct_colmap, reconstruct_dust3r, get_footage_keyframes, run_foundpose_fit,
    run_foundpose, run_full_pipeline, run_full_pipelines, get_metrics, gt_from_blender,
    reconstruct_fast3r, reconstruct_vggt
)


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
cli.add_command(reconstruct_colmap)
cli.add_command(reconstruct_dust3r)
cli.add_command(run_foundpose_fit)
cli.add_command(run_foundpose)
cli.add_command(run_full_pipeline)
cli.add_command(run_full_pipelines)
cli.add_command(get_metrics)
cli.add_command(gt_from_blender)
cli.add_command(reconstruct_fast3r)
cli.add_command(reconstruct_vggt)


if __name__ == "__main__":
    cli()
