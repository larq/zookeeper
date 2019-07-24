import click
import click_completion
import click_completion.core
import os
import functools

click_completion.init()


@click.group()
def cli():
    pass


def build_train(preload=None):
    def decorator(function):
        @click.argument("model_name")
        @click.option(
            "--dataset",
            "dataset_name",
            type=str,
            required=True,
            help="Tensorflow dataset name. See https://www.tensorflow.org/datasets/datasets for a list of available datasets",
        )
        @click.option(
            "--hparams-set",
            default="default",
            required=True,
            help="Hyperparameter set to use.",
        )
        @click.option(
            "--preprocess-fn",
            default="default",
            help="Function used to preprocess dataset.",
        )
        @click.option(
            "--hparams",
            "hparams_str",
            type=str,
            help="A comma-separated list of `name=value` hyperparameter values. This option is used to override hyperparameter settings. If a hyperparameter setting is specified by this flag then it must be a valid hyperparameter name for the model.",
        )
        @click.option("--data-dir", type=str, help="Directory with training data.")
        @click.option(
            "--data-cache",
            type=str,
            help="A directory on the filesystem to use for caching the dataset. If `--data-cache=memory`, the dataset will be cached in memory.",
        )
        @click.option(
            "--dataset-version",
            "dataset_version",
            type=str,
            help="The version of the TensorFlow dataset. See https://github.com/tensorflow/datasets/blob/master/docs/datasets_versioning.md",
        )
        @click.option(
            "--output-prefix",
            default=os.path.expanduser("~/zookeeper-logs"),
            help="Directory prefix used to save model checkpoints and logs.",
        )
        @click.option(
            "--output-dir",
            "--logdir",
            type=str,
            help="Directory containing model checkpoints. This can be used to resume model training.",
        )
        @click.option(
            "--validationset/--no-validationset",
            default=False,
            help="If you want to split a dataset which only contains a train/test into train/val/test",
        )
        @functools.wraps(function)
        def train(
            model_name,
            dataset_name,
            hparams_set,
            preprocess_fn,
            hparams_str,
            data_dir,
            data_cache,
            dataset_version,
            output_prefix,
            output_dir,
            validationset,
            **kwargs,
        ):
            from datetime import datetime
            from zookeeper import registry

            if preload:
                preload()

            dataset = registry.get_dataset(
                dataset_name,
                preprocess_fn,
                use_val_split=validationset,
                cache_dir=data_cache,
                data_dir=data_dir,
                version=dataset_version,
            )
            build_model = registry.get_model_function(model_name)
            hparams = registry.get_hparams(model_name, hparams_set)
            if hparams_str:
                hparams.parse(hparams_str)
            click.echo(hparams)

            if output_dir is None:
                time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
                output_dir = os.path.join(
                    output_prefix,
                    dataset_name,
                    model_name,
                    f"{hparams_set}_{time_stamp}",
                )

            function(build_model, dataset, hparams, output_dir, **kwargs)

        return train

    return decorator


@cli.command()
@click.argument("datasets", type=str, nargs=-1)
@click.option("--data-dir", type=str, help="Directory with training data.")
def prepare(datasets, data_dir):
    """Downloads and prepares datasets for reading."""
    import tensorflow_datasets as tfds

    for dataset in datasets:
        tfds.builder(dataset, data_dir=data_dir).download_and_prepare()
        click.secho(f"Finished preparing dataset: {dataset}", fg="green")


@cli.command(context_settings=dict(allow_extra_args=True, ignore_unknown_options=True))
@click.argument("model", required=False)
@click.option("--dataset", help="Tensorflow dataset name.")
@click.option(
    "--output-prefix",
    default=os.path.expanduser("~/zookeeper-logs"),
    help="Directory prefix used to save model checkpoints and logs.",
)
@click.option("--output-dir", "--logdir", help="Directory containing checkpoints.")
def tensorboard(model, dataset, output_prefix, output_dir):
    """Start TensorBoard to monitor model training."""
    if output_dir is None:
        output_dir = output_prefix
        if dataset:
            output_dir = os.path.join(output_dir, dataset)
            if model:
                output_dir = os.path.join(output_dir, model)
    click.secho(f"Starting TensorBoard at: {output_dir}", fg="blue")
    os.system(f"tensorboard --logdir={output_dir}")


@cli.command()
@click.argument("dataset", type=str)
@click.option(
    "--preprocess-fn", default="default", help="Function used to preprocess dataset."
)
@click.option("--data-dir", type=str, help="Directory with training data.")
@click.option(
    "--output-prefix",
    default=os.path.join(os.path.expanduser("~/zookeeper-logs"), "plots"),
    help="Directory prefix used to save plots",
)
@click.option(
    "--format", default="pdf", type=click.Choice(["png", "pdf", "ps", "eps", "svg"])
)
def plot(dataset, preprocess_fn, data_dir, output_prefix, format):
    """Plot data examples."""
    from pathlib import Path
    from zookeeper import registry, data_vis

    output_dir = Path(output_prefix).joinpath(dataset, preprocess_fn)
    output_dir.mkdir(parents=True, exist_ok=True)

    set = registry.get_dataset(dataset, preprocess_fn, data_dir=data_dir)
    figs = data_vis.plot_all_examples(set)
    for fig, filename in zip(figs, ("raw", "train", "eval")):
        fig.savefig(f"{output_dir.joinpath(filename).absolute()}.{format}")


@cli.command()
@click.option(
    "--append/--overwrite", help="Append the completion code to the file", default=None
)
@click.option(
    "-i", "--case-insensitive/--no-case-insensitive", help="Case insensitive completion"
)
@click.argument(
    "shell",
    required=False,
    type=click_completion.DocumentedChoice(click_completion.core.shells),
)
@click.argument("path", required=False)
def install_completion(append, case_insensitive, shell, path):
    """Install shell completion."""
    extra_env = (
        {"_CLICK_COMPLETION_COMMAND_CASE_INSENSITIVE_COMPLETE": "ON"}
        if case_insensitive
        else {}
    )
    shell, path = click_completion.core.install(
        shell=shell, path=path, append=append, extra_env=extra_env
    )
    click.secho(f"{shell} completion installed in {path}.", fg="green")


if __name__ == "__main__":  # pragma: no cover
    cli()
