import pytest
from click import testing

from zookeeper.core.cli import cli
from zookeeper.core.component import component
from zookeeper.core.factory import factory
from zookeeper.core.field import ComponentField, Field
from zookeeper.core.task import task


@pytest.fixture
def test_task_runner():
    @task
    class TestTask:
        a: int = Field()
        b: str = Field("foo")
        c: bool = Field(False)

        def run(self):
            print(self.a, self.b, self.c)

    yield testing.CliRunner(mix_stderr=False)

    # Clear existing commands.
    cli.commands = dict()


def test_command_aliases(test_task_runner):
    # We should be able to invoke commands with strings that resolve to the same
    # camel-case.
    assert test_task_runner.invoke(cli, ["test_task", "a=5"]).exit_code == 0
    assert test_task_runner.invoke(cli, ["TestTask", "a=5"]).exit_code == 0
    assert test_task_runner.invoke(cli, ["Test_Task", "a=5"]).exit_code == 0
    assert test_task_runner.invoke(cli, ["testTask", "a=5"]).exit_code == 0
    assert test_task_runner.invoke(cli, ["NotTestTask", "a=5"]).exit_code != 0


def test_pass_param_values(test_task_runner):
    # We should be able to pass parameter values through the CLI.
    result = test_task_runner.invoke(cli, ["test_task", "a=5"])
    assert result.exit_code == 0
    assert result.output == "5 foo False\n"


def test_param_key_valid_characters():
    # We should be able to pass keys with underscores and full stops and
    # capitals.

    @component
    class Child:
        x_Y_z: float = Field(0.0)

    @task
    class ParentTask:
        a: int = Field(2)
        child: Child = ComponentField(Child)

        def run(self):
            print(self.a, self.child.x_Y_z)

    runner = testing.CliRunner(mix_stderr=False)
    result = runner.invoke(cli, ["ParentTask", "a=5", "child.x_Y_z=1.0"])
    assert result.exit_code == 0


def test_param_key_invalid_characters(test_task_runner):
    # Keys with invalid characters such as '-' or '@' should not be accepted.
    result = test_task_runner.invoke(cli, ["test_task", "a=5", "x-y=1.0"])
    assert result.exit_code == 2
    result = test_task_runner.invoke(cli, ["test_task", "a=5", "x@y=1.0"])
    assert result.exit_code == 2


def test_override_param_values(test_task_runner):
    # We should be able to override existing parameter values through the CLI.
    result = test_task_runner.invoke(cli, ["test_task", "a=5", "b=bar", "c=True"])
    assert result.exit_code == 0
    assert result.output == "5 bar True\n"


def test_override_param_complex_string(test_task_runner):
    # We should be able to pass complex strings, including paths.
    result = test_task_runner.invoke(
        cli, ["test_task", "a=5", "b=https://some-path/foo/bar@somewhere"]
    )
    assert result.exit_code == 0
    assert result.output == "5 https://some-path/foo/bar@somewhere False\n"


def test_boolean_flag_syntax(test_task_runner):
    # We should be able to use a shorthand for setting boolean flags.
    result = test_task_runner.invoke(cli, ["test_task", "a=5", "--c"])
    assert result.exit_code == 0
    assert result.output == "5 foo True\n"
    result = test_task_runner.invoke(cli, ["test_task", "a=5", "--no-c"])
    assert result.exit_code == 0
    assert result.output == "5 foo False\n"


def test_component_and_factory_override():
    class Base:
        name = "abstract_base"

    @component
    class BaseComponent(Base):
        name = "component_base"

    @factory
    class BaseFactory:
        def build(self) -> Base:
            value = Base()
            value.name = "factory_base"
            return value

    @task
    class TestTask:
        base: Base = ComponentField()

        def run(self):
            print(self.base.name)

    runner = testing.CliRunner(mix_stderr=False)

    # Errors with no provided `base` value.
    result = runner.invoke(cli, ["test_task"])
    assert result.exit_code == 1

    # Succeeds with @component.
    result = runner.invoke(cli, ["test_task", "base=BaseComponent"])
    assert result.exit_code == 0
    assert result.output == "component_base\n"

    # Succeeds with @factory.
    result = runner.invoke(cli, ["test_task", "base=BaseFactory"])
    assert result.exit_code == 0
    assert result.output == "factory_base\n"

    # Clear existing commands.
    cli.commands = dict()
