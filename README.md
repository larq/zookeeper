# Zookeeper

[![GitHub Actions](https://github.com/larq/zookeeper/workflows/Unittest/badge.svg)](https://github.com/larq/zookeeper/actions?workflow=Unittest) [![Codecov](https://img.shields.io/codecov/c/github/larq/zookeeper)](https://codecov.io/github/larq/zookeeper?branch=main) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zookeeper.svg)](https://pypi.org/project/zookeeper/) [![PyPI](https://img.shields.io/pypi/v/zookeeper.svg)](https://pypi.org/project/zookeeper/) [![PyPI - License](https://img.shields.io/pypi/l/zookeeper.svg)](https://github.com/plumerai/zookeeper/blob/main/LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

A small library for configuring modular applications.

### Installation

```console
pip install zookeeper
```

### Components

The fundamental building blocks of Zookeeper are components. The
[`@component`](zookeeper/component.py) decorator is used to turn classes into
components. These component classes can have configurable fields, which are
declared with the `Field` constructor and class-level type annotations. Fields
can be created with or without default values. Components can also be nested,
with `ComponentField`s, such that child componenents can access the field values
defined on their parents.

For example:

```python
from zookeeper import component

@component
class ChildComponent:
    a: int = Field()                          # An `int` field with no default set
    b: str = Field("foo")                     # A `str` field with default value `"foo"`

@component
class ParentComponent:
    a: int = Field()                          # The same `int` field as the child
    child: ChildComponent = ComponentField()  # A nested component field, of type `ChildComponent`
```

After instantiation, components can be 'configured' with a configuration
dictionary, containing values for a tree of nested fields. This process
automatically injects the correct values into each field.

If a child sub-component declares a field which already exists in some
containing ancestor component, then it will pick up the value that's set on the
parent, unless a 'scoped' value is set on the child.

For example:

```
from zookeeper import configure

p = ParentComponent()

configure(
    p,
    {
        "a": 5,
        "child.a": 4,
    }
)

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

The [`@task`](zookeeper/task.py) decorator is used to define Zookeeper tasks and
can be applied to any class that implements an argument-less `run` method. Such
tasks can be run through the Zookeeper CLI, with parameter values passed in
through CLI arguments (`configure` is implicitly called).

For example:

```python
from zookeeper import cli, task

@task
class UseChildA:
    parent: ParentComponent = ComponentField()
    def run(self):
        print(self.parent.child.a)

@task
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
MNIST. This example can be easily adapted to other Larq or Keras models and
other datasets.
