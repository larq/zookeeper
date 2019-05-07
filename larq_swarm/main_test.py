from larq_swarm.main import cli
from click.testing import CliRunner
import fixtures


def test_cli():
    assert fixtures
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
