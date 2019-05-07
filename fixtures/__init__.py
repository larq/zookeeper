import click
from larq_swarm import (
    register_model,
    register_hparams,
    register_preprocess,
    cli,
    build_train,
    HParams,
    Dataset,
)


@register_preprocess("mnist")
def default(image):
    return image


@register_model
def foo(hparams, dataset):
    return "foo-model"


@register_hparams(foo)
def bar():
    return HParams(baz=3, baz_overwrite=0)


@cli.command()
@click.option("--custom-opt", type=str, required=True)
@build_train
def train(build_model, dataset, hparams, output_dir, epochs, custom_opt):
    assert isinstance(hparams, HParams)
    assert isinstance(dataset, Dataset)
    assert isinstance(output_dir, str)
    assert isinstance(epochs, int)
    assert isinstance(custom_opt, str)

    model = build_model(hparams, dataset)
    assert model == "foo-model"
    assert dataset.dataset_name == "mnist"
    assert hparams.baz == 3
    assert hparams.baz_overwrite == 42
    assert custom_opt == "passed"
    print("TESTS PASSED")


if __name__ == "__main__":
    cli()
