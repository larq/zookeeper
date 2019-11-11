# Zookeeper

[![GitHub Actions](https://github.com/larq/zookeeper/workflows/Unittest/badge.svg)](https://github.com/larq/zookeeper/actions?workflow=Unittest) [![Codecov](https://img.shields.io/codecov/c/github/larq/zookeeper)](https://codecov.io/github/larq/zookeeper?branch=master) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zookeeper.svg)](https://pypi.org/project/zookeeper/) [![PyPI](https://img.shields.io/pypi/v/zookeeper.svg)](https://pypi.org/project/zookeeper/) [![PyPI - License](https://img.shields.io/pypi/l/zookeeper.svg)](https://github.com/plumerai/zookeeper/blob/master/LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black) [![Join the community on Spectrum](https://withspectrum.github.io/badge/badge.svg)](https://spectrum.chat/larq)

A small library for configuring modular applications.

### Installation

```console
pip install zookeeper
```

### Components

The fundamental building block of Zookeeper is a
[`Component`](zookeeper/component.py). `Component` subclasses can have
configurable parameters, which are declared using class-level type annotations
(in a similar way to [Python
dataclasses](https://docs.python.org/3/library/dataclasses.html)). These
parameters can be Python objects or nested sub-components, and need not be set
with a default value.

For example:
```python
from zookeeper import Component

class ChildComponent(Component):
    a: int                  # An `int` parameter, with no default set
    b: str = "foo"          # A `str` parameter, which by default will be `foo`

class ParentComponent(Component):
    a: int                  # The same `int` parameter as the child
    child: ChildComponent   # A nested component parameter, of type `ChildComponent`
```

After instantiation, components can be 'configured' with a configuration
dictionary, containing values for a tree of nested parameters. This process
automatically injects the correct values into each parameter.

If a child sub-component declares a parameter which already exists in some
containing parent, then it will pick up the value that's set on the parent,
unless a 'scoped' value is set on the child.

For example:
```
p = ParentComponent()

p.configure({
    "a": 5,
    "child.a": 4,
})

>>> 'ChildComponent' is the only concrete component class that satisfies the type
>>> of the annotated parameter 'ParentComponent.child'. Using an instance of this
>>> class by default.

print(p)

>>> ParentComponent(
>>>     a = 5,
>>>     child = ChildComponent(
>>>         a = 4,
>>>         b = "foo"
>>>     )
>>> )
```

### Tasks and the CLI

The best way to define runnable tasks with Zookeeper is to subclass
[`Task`](zookeeper/task.py) and override the `run` method.

Zookeeper provides a small mechanism to run tasks from a CLI, using the
decorator `@add_task_to_cli`. The CLI will automatically instantiate the task
and call `configure()`, passing in configuration parsed from command line
arguments.

For example:
```python
from zookeeper import Task
from zookeeper.cli import add_task_to_cli, cli

@add_task_to_cli
class UseChildA(Task):
    parent: ParentComponent

    def run(self):
        print(self.parent.child.a)

@add_task_to_cli
class UseParentA(UseChildA):
    def run(self):
        print(self.parent.a)

if __name__ == "__main__":
    cli()
```

Running the above file then gives a nice CLI interface:
```
python test.py use_child_a
>>> ValueError: No configuration value found for annotated parameter 'UseChildA.parent.a' of type 'int'.

python test.py use_child_a a=5
>>> 5

python test.py use_child_a a=5 child.a=3
>>> 3

python test.py use_parent_a a=5 child.a=3
>>> 5
```

### Using Zookeeper to define Larq or Keras experiments

See [examples/larq_experiment.py](examples/larq_experiment.py) for an example of
how to use Zookeeper to define all the necessary components (dataset,
preprocessing, and model) of a Larq experiment: training a BinaryNet on
CIFAR-10. This example can be easily adapted to other Larq or Keras models and
other datasets.
