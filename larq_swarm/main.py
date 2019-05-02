import click
import os
from datetime import datetime
import logging

log = logging.getLogger(name=__name__)
log.setLevel(logging.INFO)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("model_name")
@click.option(
    "--dataset",
    "dataset_name",
    type=str,
    required=True,
    help="Tensorflow dataset name. See https://www.tensorflow.org/datasets/datasets for a list of available datasets",
)
@click.option(
    "--hparams-set", default="default", required=True, help="Hyperparameter set to use."
)
@click.option("--epochs", default=100, help="The number of epochs to run training for.")
@click.option(
    "--initial-epoch", default=0, help="Initial epoch (for learning rate etc.)."
)
@click.option(
    "--preprocess-fn", default="default", help="Function used to preprocess dataset."
)
@click.option(
    "--hparams",
    "hparams_str",
    type=str,
    help="A comma-separated list of `name=value` hyperparameter values. This option is used to override hyperparameter settings. If a hyperparameter setting is specified by this flag then it must be a valid hyperparameter name for the model.",
)
@click.option("--train-fn", type=str, default="train")
@click.option("--data-dir", type=str, help="Directory with training data.")
@click.option(
    "--output-prefix",
    default=os.path.expanduser("~/larq-swarm-logs"),
    help="Directory prefix used to save model checkpoints and logs.",
)
@click.option(
    "--output-dir",
    type=str,
    help="Directory containing model checkpoints. This can be used to resume model training.",
)
@click.option(
    "--pretrain-dir",
    type=str,
    help="Directory containing pretrained checkpoint used to initialize the network.",
)
@click.option("--tensorboard/--no-tensorboard", default=True)
@click.option(
    "--validationset/--no-validationset",
    default=False,
    help="If you want to split a dataset which only contains a train/test into train/val/test",
)
@click.option(
    "--validationset/--no-validationset",
    default=False,
    help="If you want to split a dataset which only contains a train/test into train/val/test",
)
@click.option(
    "--usr-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=os.getcwd(),
)
def train(
    model_name,
    dataset_name,
    hparams_set,
    epochs,
    initial_epoch,
    preprocess_fn,
    hparams_str,
    train_fn,
    data_dir,
    output_prefix,
    output_dir,
    pretrain_dir,
    tensorboard,
    validationset,
    usr_dir,
):
    from larq_swarm import registry, utils

    utils.import_usr_dir(usr_dir)
    dataset = registry.get_dataset(dataset_name, preprocess_fn, validationset, data_dir)
    hparams = registry.get_hparams(model_name, hparams_set)
    if hparams_str:
        hparams.parse(hparams_str)

    build_model = registry.get_model_function(model_name)
    train_fn = registry.get_train_function(train_fn)

    if output_dir is None:
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = os.path.join(
            output_prefix, dataset_name, model_name, f"{hparams_set}_{time_stamp}"
        )

    train_fn(
        build_model,
        dataset,
        hparams,
        output_dir,
        epochs,
        initial_epoch,
        pretrain_dir,
        tensorboard,
    )


@cli.command()
@click.argument("datasets", type=str, nargs=-1)
@click.option("--data-dir", type=str, help="Directory with training data.")
def prepare(datasets, data_dir):
    from larq_models.data import get_dataset

    for dataset in datasets:
        get_dataset(dataset, "default", data_dir)
        print("Finished preparing dataset:", dataset)


if __name__ == "__main__":
    cli()
