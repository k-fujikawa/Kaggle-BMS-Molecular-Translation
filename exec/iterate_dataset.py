import click
from omegaconf import OmegaConf
from tqdm.auto import tqdm

import nncomp
import nncomp.registry as R
import nncomp_molecule  # NOQA


@click.command()
@click.argument("config")
@click.argument("overrides", nargs=-1)
def main(config, overrides):
    config = nncomp.config.load_config(config)
    OmegaConf.set_struct(config, True)
    config.merge_with_dotlist(overrides)

    if config.debug:
        config.name = "debug"
        config.inputs.train = "/work/output/preprocess/debug/train.pkl"
        config.num_workers = 0

    nncomp.utils.set_seed(config["seed"])
    config = OmegaConf.to_container(config, resolve=True)
    print(OmegaConf.to_yaml(config))

    experiment = R.ExperimentRegistry.get_from_params(
        **config["experiment_params"],
        config=config,
    )
    loader = experiment._get_loader(stage="fold=0", name = "train")
    list(tqdm(loader))


if __name__ == "__main__":
    main()
