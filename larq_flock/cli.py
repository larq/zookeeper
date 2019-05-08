import click
import os
from datetime import datetime


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
        "--output-prefix",
        default=os.path.expanduser("~/larq-flock-logs"),
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
    def train(
        model_name,
        dataset_name,
        hparams_set,
        epochs,
        preprocess_fn,
        hparams_str,
        data_dir,
        output_prefix,
        output_dir,
        validationset,
        **kwargs,
    ):
        from larq_flock import registry

        dataset = registry.get_dataset(
            dataset_name, preprocess_fn, validationset, data_dir
        )
        hparams = registry.get_hparams(model_name, hparams_set)
        if hparams_str:
            hparams.parse(hparams_str)

        build_model = registry.get_model_function(model_name)

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
    default=os.path.expanduser("~/larq-flock-logs"),
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


if __name__ == "__main__":  # pragma: no cover
    cli()
