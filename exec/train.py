import shutil
import yaml
from datetime import datetime
from pathlib import Path

import click
from omegaconf import OmegaConf

import nncomp
import nncomp.registry as R
import nncomp_molecule  # NOQA


@click.command()
@click.argument("config")
@click.argument("overrides", nargs=-1)
def main(config, overrides):
    config = nncomp.config.load_config(config)
    OmegaConf.set_struct(config, True)
    overrides = OmegaConf.from_dotlist(overrides)
    config = OmegaConf.merge(config, overrides)
    if config.bench:
        config.name = "[bench]" + config.name
        config.wandb.tags.append("bench")
        for key, path in config["inputs"].items():
            config["inputs"][key] = str(Path(path).with_suffix(".bench.pkl"))
    if config.debug:
        config.name = "debug"
        config.num_workers = 0
        config.notify_slack = False
        for key, path in config["inputs"].items():
            config["inputs"][key] = str(Path(path).with_suffix(".debug.pkl"))
    config = OmegaConf.merge(config, overrides)

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    config.experiment_id = f"{config.name}{config.suffix}_{now}"
    if config.resume is True:
        config.resume_dir = str(config.outdir)
    if config.resume_dir is not None:
        resume_dir = sorted(Path(config.resume_dir).glob("fold=*"))[-1]
        config.resume = str(resume_dir / "checkpoints/last_full.pth")
        with open(resume_dir.parent / "config.yml") as f:
            _config = yaml.safe_load(f)
            config.experiment_id = _config["experiment_id"]

    nncomp.utils.set_seed(config["seed"])
    outdir = Path(config["outdir"])

    is_continue = False
    if config["resume_dir"] is not None:
        is_continue = Path(config["resume_dir"]).resolve() == outdir.resolve()
    if outdir.exists() and not is_continue:
        if config["debug"] or click.confirm(f"Delete directory: ${outdir}"):
            shutil.rmtree(outdir)
    if config["resume_dir"] is not None and not is_continue:
        shutil.copytree(config["resume_dir"], config["outdir"])

    outdir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=outdir / "config.yml")

    config = OmegaConf.to_container(config, resolve=True)
    print(OmegaConf.to_yaml(config))
    experiment = R.ExperimentRegistry.get_from_params(
        **config["experiment_params"],
        config=config,
    )
    runner = R.RunnerRegistry.get_from_params(
        **config["runner_params"],
        device=nncomp.utils.get_device(config["device"]),
    )
    runner.run_experiment(experiment)


if __name__ == "__main__":
    main()
