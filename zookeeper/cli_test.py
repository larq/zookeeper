import pytest
from click.testing import CliRunner

from zookeeper import Task
from zookeeper.cli import add_task_to_cli, cli


@pytest.fixture
def test_task():
    # We have to define `TestTask` inside a pytest fixture so that it gets
    # reinstantiated for every test.

    @add_task_to_cli
    class TestTask(Task):
        a: int
        b: str = "foo"

        def run(self):
            print(self.a, self.b)

    return None


runner = CliRunner(mix_stderr=False)


def test_pass_param_values(test_task):
    # We should be able to pass parameter values through the CLI.
    result = runner.invoke(cli, ["test_task", "a=5"])
    assert result.exit_code == 0
    assert result.output == "5 foo\n"


def test_param_key_valid_characters(test_task):
    # We should be able to pass keys with underscores and full stops and
    # capitals. It's okay here that the param with name `x.y_z.A` doesn't
    # actually exist.
    result = runner.invoke(cli, ["test_task", "a=5", "x.y_z.A=1.0"])
    assert result.exit_code == 0


def test_param_key_invalid_characters(test_task):
    # Keys with invalid characters such as '-' or '@' should not be accepted.
    result = runner.invoke(cli, ["test_task", "a=5", "x-y=1.0"])
    assert result.exit_code == 2
    result = runner.invoke(cli, ["test_task", "a=5", "x@y=1.0"])
    assert result.exit_code == 2


def test_override_param_values(test_task):
    # We should be able to override existing parameter values through the CLI.
    result = runner.invoke(cli, ["test_task", "a=5", "b=bar"])
    assert result.exit_code == 0
    assert result.output == "5 bar\n"


def test_override_param_complex_string(test_task):
    # We should be able to pass complex strings, including paths.
    result = runner.invoke(
        cli, ["test_task", "a=5", "b=https://some-path/foo/bar@somewhere"]
    )
    assert result.exit_code == 0
    assert result.output == "5 https://some-path/foo/bar@somewhere\n"
