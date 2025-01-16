import click


@click.group()
def cli():
    pass


@cli.command()
def datagen_train_run():
    from burybarrel.scripts import datagen_occ, train_barrelnet

    datagen_occ.run()
    train_barrelnet.run()


@cli.command()
def run_pointnet_inference():
    from burybarrel.scripts import run_pointnet_inf

    run_pointnet_inf.run()


@cli.command()
@click.option(
    "-c",
    "--cfg",
    "footage_cfg_path",
    default="configs/footage.yaml",
    show_default=True,
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("-n", "--name", "footage_name", required=True, type=click.STRING, help="Key name inside config YAML")
@click.option(
    "--fps", "fps", required=True, default=25, show_default=True, type=click.INT
)
@click.option("--step", "step", required=True, type=click.INT, help="Number of frames between each image")
@click.option(
    "--crop/--no-crop", "crop", is_flag=True, default=True, show_default=True, type=click.BOOL
)
@click.option(
    "--contrast/--no-contrast",
    "increase_contrast",
    is_flag=True,
    default=True,
    show_default=True,
    type=click.BOOL,
)
@click.option("--navpath", "navpath", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--denoisedepth/--no-denoisedepth",
    "denoise_depth",
    is_flag=True,
    default=True,
    show_default=True,
    type=click.BOOL,
    help="Only used when navpath is provided; naively denoises depth data"
)
def get_footage_keyframes(
    footage_cfg_path,
    footage_name,
    fps,
    step,
    navpath,
    crop,
    increase_contrast,
    denoise_depth,
):
    from burybarrel.scripts import get_footage_keyframes

    get_footage_keyframes.run(
        footage_cfg_path,
        footage_name,
        step,
        navpath=navpath,
        crop=crop,
        fps=fps,
        increase_contrast=increase_contrast,
        denoise_depth=denoise_depth,
    )


if __name__ == "__main__":
    cli()
