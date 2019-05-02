from larq_swarm.main import cli
from click.testing import CliRunner


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
            "--usr-dir",
            "fixtures",
            "--dataset",
            "mnist",
            "--hparams",
            "baz_overwrite=42",
        ],
    )
    assert result.exit_code == 0
    assert result.output == "TESTS PASSED\n"
