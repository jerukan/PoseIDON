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
@click.option("-i", "--input", "input_vid", required=True, type=click.Path(exists=True, dir_okay=False), help="input video path")
@click.option("-o", "--output", "output_dir", required=False, type=click.Path(file_okay=False), help="output dir")
@click.option("-t", "--time", "start_time", required=False, type=click.STRING, help="start time of the video")
@click.option("--tz", "timezone", required=False, type=click.STRING, help="timezone")
@click.option(
    "--fps", "fps", required=True, default=25, show_default=True, type=click.INT
)
@click.option("--step", "step", required=True, type=click.INT, help="Number of frames between each image")
@click.option(
    "--crop/--no-crop", "crop", is_flag=True, default=True, show_default=True, type=click.BOOL
)
@click.option(
    "--contrast",
    "increase_contrast",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="If provided, increases contrast of the images",
)
@click.option("--navpath", "navpath", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--denoisedepth",
    "denoise_depth",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Only used when navpath is provided; naively denoises depth data"
)
def get_footage_keyframes(
    input_vid,
    output_dir,
    start_time,
    timezone,
    fps,
    step,
    navpath,
    crop,
    increase_contrast,
    denoise_depth,
):
    from burybarrel.scripts import get_footage_keyframes

    get_footage_keyframes.run(
        input_vid,
        step,
        output_dir=output_dir,
        start_time=start_time,
        timezone=timezone,
        navpath=navpath,
        crop=crop,
        fps=fps,
        increase_contrast=increase_contrast,
        denoise_depth=denoise_depth,
    )


@cli.command()
@click.option(
    "-i",
    "--imgdir",
    "imgdir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option("-p", "--prompt", "text_prompt", required=True, type=click.STRING)
@click.option(
    "-o",
    "--outdir",
    "outdir",
    required=True,
    type=click.Path(file_okay=False),
)
@click.option(
    "--boxthresh",
    "box_threshold",
    default=0.3,
    show_default=True,
    type=click.FLOAT,
)
@click.option(
    "--textthresh",
    "text_threshold",
    default=0.25,
    show_default=True,
    type=click.FLOAT,
)
@click.option(
    "--closekernel",
    "closekernelsize",
    default=0,
    type=click.INT,
    show_default=True,
    help="n x n kernel size for morphological closing operation; set to 0 for no closing",
)
def create_masks(imgdir, text_prompt, outdir, box_threshold, text_threshold, closekernelsize):
    from burybarrel.scripts import create_masks

    create_masks.run(
        imgdir, text_prompt, outdir, box_threshold=box_threshold, text_threshold=text_threshold, closekernelsize=closekernelsize
    )


@cli.command()
@click.option(
    "-m",
    "--modelpath",
    "model_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-i",
    "--imgdir",
    "imgdir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "-o",
    "--outdir",
    "outdir",
    required=True,
    type=click.Path(file_okay=False),
)
@click.option(
    "-d",
    "--device",
    "device",
    type=click.STRING,
)
def reconstruct_dust3r(model_path, imgdir, outdir, device):
    from burybarrel.scripts import reconstruct_dust3r

    reconstruct_dust3r.run(model_path, imgdir, outdir, device=device)


@cli.command()
@click.option(
    "-i",
    "--imgdir",
    "imgdir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
)
@click.option(
    "-o",
    "--outdir",
    "outdir",
    required=True,
    type=click.Path(file_okay=False),
)
@click.option(
    "--sparse",
    "sparse",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Whether to run sparse reconstruction",
)
@click.option(
    "--dense",
    "dense",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Whether to run dense + mesh + texture reconstruction (requires sparse to be run first)",
)
@click.option(
    "--overwrite",
    "overwrite",
    is_flag=True,
    default=False,
    type=click.BOOL,
    help="Overwrite existing COLMAP database if it exists (it complains by default)",
)
def reconstruct_colmap(imgdir, outdir, sparse, dense, overwrite):
    from burybarrel.scripts import reconstruct_colmap

    reconstruct_colmap.run(imgdir, outdir, sparse=sparse, dense=dense, overwrite=overwrite)


if __name__ == "__main__":
    cli()
