from larq_swarm.main import cli
from click.testing import CliRunner


def test_cli():
    result = CliRunner(mix_stderr=False).invoke(
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
        ],
    )
    assert result.exit_code == 0
    assert result.output == "TESTS PASSED\n"
