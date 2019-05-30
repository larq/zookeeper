import click
import os
from pathlib import Path
from datetime import datetime
from functools import wraps


@click.group()
def cli():
    pass


def build_train(function):
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
        "--epochs", default=100, help="The number of epochs to run training for."
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
    @wraps(function)
    def train(
        model_name,
        dataset_name,
        hparams_set,
        epochs,
        preprocess_fn,
        hparams_str,
        data_dir,
        data_cache,
        output_prefix,
        output_dir,
        validationset,
        **kwargs,
    ):
        from zookeeper import registry

        dataset = registry.get_dataset(
            dataset_name,
            preprocess_fn,
            use_val_split=validationset,
            cache_dir=data_cache,
            data_dir=data_dir,
        )
        build_model = registry.get_model_function(model_name)
        hparams = registry.get_hparams(model_name, hparams_set)
        if hparams_str:
            hparams.parse(hparams_str)
        click.echo(hparams)

        if output_dir is None:
            time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_dir = os.path.join(
                output_prefix, dataset_name, model_name, f"{hparams_set}_{time_stamp}"
            )

        function(build_model, dataset, hparams, output_dir, epochs, **kwargs)

    return train


@cli.command()
@click.argument("datasets", type=str, nargs=-1)
@click.option("--data-dir", type=str, help="Directory with training data.")
def prepare(datasets, data_dir):
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
    from zookeeper import registry, data_vis

    output_dir = Path(output_prefix).joinpath(dataset, preprocess_fn)
    output_dir.mkdir(parents=True, exist_ok=True)

    set = registry.get_dataset(dataset, preprocess_fn, data_dir=data_dir)
    figs = data_vis.plot_all_examples(set.load_split(set.train_split), set.map_fn)
    for fig, filename in zip(figs, ("raw", "train", "eval")):
        fig.savefig(output_dir.joinpath(filename).absolute(), format=format)


if __name__ == "__main__":  # pragma: no cover
    cli()
