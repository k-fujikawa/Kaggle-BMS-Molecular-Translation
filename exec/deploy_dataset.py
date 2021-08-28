from pathlib import Path

import click

import nncomp.io.upload


@click.command()
@click.argument("deploydir")
@click.option(
    "--dir-mode",
    type=click.Choice(["zip", "skip", "tar"]),
    default="zip",
)
def main(deploydir, dir_mode):
    deploydir = Path(deploydir)
    nncomp.io.upload.create_or_update_kaggle_dataset(
        srcdir=deploydir,
        title=deploydir.name,
        dir_mode=dir_mode,
        delete_old_versions=False,
    )


if __name__ == '__main__':
    main()
