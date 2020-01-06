from click import testing

from zookeeper.core.cli import cli
from zookeeper.core.task import task


@task
class TestTask:
    a: int
    b: str = "foo"

    def run(self):
        print(self.a, self.b)


runner = testing.CliRunner(mix_stderr=False)


def test_command_aliases():
    # We should be able to invoke commands with strings that resolve to the same
    # camel-case.
    assert runner.invoke(cli, ["test_task", "a=5"]).exit_code == 0
    assert runner.invoke(cli, ["TestTask", "a=5"]).exit_code == 0
    assert runner.invoke(cli, ["Test_Task", "a=5"]).exit_code == 0
    assert runner.invoke(cli, ["testTask", "a=5"]).exit_code == 0
    assert runner.invoke(cli, ["NotTestTask", "a=5"]).exit_code != 0


def test_pass_param_values():
    # We should be able to pass parameter values through the CLI.
    result = runner.invoke(cli, ["test_task", "a=5"])
    assert result.exit_code == 0
    assert result.output == "5 foo\n"


def test_param_key_valid_characters():
    # We should be able to pass keys with underscores and full stops and
    # capitals. It's okay here that the param with name `x.y_z.A` doesn't
    # actually exist.
    result = runner.invoke(cli, ["test_task", "a=5", "x.y_z.A=1.0"])
    assert result.exit_code == 0


def test_param_key_invalid_characters():
    # Keys with invalid characters such as '-' or '@' should not be accepted.
    result = runner.invoke(cli, ["test_task", "a=5", "x-y=1.0"])
    assert result.exit_code == 2
    result = runner.invoke(cli, ["test_task", "a=5", "x@y=1.0"])
    assert result.exit_code == 2


def test_override_param_values():
    # We should be able to override existing parameter values through the CLI.
    result = runner.invoke(cli, ["test_task", "a=5", "b=bar"])
    assert result.exit_code == 0
    assert result.output == "5 bar\n"


def test_override_param_complex_string():
    # We should be able to pass complex strings, including paths.
    result = runner.invoke(
        cli, ["test_task", "a=5", "b=https://some-path/foo/bar@somewhere"]
    )
    assert result.exit_code == 0
    assert result.output == "5 https://some-path/foo/bar@somewhere\n"
