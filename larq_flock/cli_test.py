from larq_flock import registry, cli, build_train, HParams, data
from click.testing import CliRunner
import click


@registry.register_preprocess("mnist")
def default(image):
    return image


@registry.register_model
def foo(hparams, dataset):
    return "foo-model"


@registry.register_hparams(foo)
def bar():
    return HParams(baz=3, baz_overwrite=0)


@cli.command()
@click.option("--custom-opt", type=str, required=True)
@build_train
def train(build_model, dataset, hparams, output_dir, epochs, custom_opt):
    assert isinstance(hparams, HParams)
    assert isinstance(dataset, data.Dataset)
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


def test_cli():
    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(cli, ["prepare", "mnist"])
    assert result.exit_code == 0

    result = runner.invoke(
        cli,
        [
            "train",
            "foo",
            "--hparams-set",
            "bar",
            "--dataset",
            "mnist",
            "--hparams",
            "baz_overwrite=42",
            "--custom-opt",
            "passed",
        ],
    )
    assert result.exit_code == 0
    assert result.output == "TESTS PASSED\n"
