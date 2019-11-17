import re
from abc import ABC, abstractmethod
from typing import List
from unittest.mock import patch

import pytest
from click import unstyle

from zookeeper.component import Component


# Specify this as a fixture because we want the class to be re-defined each
# time. This is because we want class-level default attributes to be re-defined
# and re-initialised.
@pytest.fixture
def Parent():
    class AbstractGrandChild(Component, ABC):
        a: int
        b: str
        c: List[float]

        @abstractmethod
        def __call__(self):
            pass

    class GrandChild1(AbstractGrandChild):
        def __call__(self):
            return f"grand_child_1_{self.b}_{self.c}"

    class GrandChild2(AbstractGrandChild):
        def __call__(self):
            return f"grand_child_2_{self.a}"

    class Child(Component):
        b: str = "bar"

        grand_child: AbstractGrandChild = GrandChild1()

        def __call__(self):
            return f"child_{self.b}_{self.grand_child()}"

    class Parent(Component):
        a: int
        b: str

        child: Child

        def __call__(self):
            return f"root_{self.a}_{self.child()}"

    return Parent


def test_override_init_error():
    # Defining a subclass which overrides `__init__` should raise a ValueError.

    with pytest.raises(ValueError, match=r"^Overriding `__init__` in component"):

        class C(Component):
            def __init__(self, c, **kwargs):
                super().__init__(**kwargs)
                self.c = c


def test_init(Parent):
    # Kwargs passed to `__init__` corresponding to annotated parameter should
    # set those values, overriding the defaults.

    p = Parent(a=5)
    assert p.a == 5

    # Kwargs not corresponding to an annotated parameter should cause a
    # ValueError to be raised.
    with pytest.raises(
        ValueError,
        match=r"Argument 'd' passed to `__init__` does not correspond to any annotation",
    ):
        p = Parent(d="baz")


def test_configure_non_interactive_missing_param(Parent):
    # Hydrating without configuring a value for an attribute without a default
    # should raise a ValueError.

    p = Parent()
    p_conf = {"a": 5}
    with pytest.raises(
        ValueError, match=r"^No configuration value found for annotated parameter"
    ):
        p.configure(p_conf, name="parent")


def test_configure_override_values(Parent):
    # A configured instance should have its attributes corrected overriden.

    p = Parent()
    p_conf = {"a": 10, "b": "foo", "c": [1.5, -1.2]}

    p.configure(p_conf, name="parent")

    # `a` should be correctly overrriden, `b` should take its default value.
    assert p.a == 10
    assert p.b == "foo"


def test_configure_scoped_override(Parent):
    # Configuration values should be correctly scoped.

    p = Parent()
    p_conf = {
        "a": 10,
        "child.a": 15,
        "b": "foo",
        "child.grand_child.b": "baz",
        "c": [1.5, -1.2],
        "child.c": [-17.2],
        "child.grand_child.c": [0, 4.2],
    }

    p.configure(p_conf, name="parent")

    # The parent `p` should have the value `a` = 10. Even though a config value
    # is declared for its scope, `p.child` should have no `a` value set, as it
    # doesn't declare `a` as a dependency. Despite this, `p.child.grand_child`
    # should get the value `a` = 15, as it lives inside the configuration scope
    # of its parent, `p.child`.
    assert p.a == 10
    assert not hasattr(p.child, "a")
    assert p.child.grand_child.a == 15

    # `b` is declared as a dependency at all three levels. The `baz` value
    # should be scoped only to the grandchild, so `foo` will apply to both
    # higher levels.
    assert p.b == "foo"
    assert p.child.b == "foo"
    assert p.child.grand_child.b == "baz"

    # `c` is declared as a dependency only in the grandchild. The more specific
    # scopes override the more general.
    assert p.child.grand_child.c == [0, 4.2]


def test_configure_one_possible_component():
    # If there's only a single defined, non-abstract class that satisfies a
    # declared sub-component depency of a component, then we expect `configure`
    # to instantiate an instance of this class by default without prompting, but
    # also warn that it has done so.
    class A(Component):
        def __call__(self):
            return "hello world"

    class Parent(Component):
        a: A

        def __call__(self):
            return self.a.__call__()

    p = Parent()

    with patch("zookeeper.component.print_formatted_text") as print_formatted_text:
        p.configure({})

    print_formatted_text.assert_called_once()
    assert len(print_formatted_text.call_args[0]) == 1
    assert re.search(
        r" is the only concrete component class that satisfies the type of the "
        "annotated parameter 'Parent.a'. Using an instance of this class by default.$",
        print_formatted_text.call_args[0][0],
    )


def test_configure_interactive_prompt_for_missing_value(Parent):
    # Configure with all configuration values specified apart from `c`. When
    # running in interactive mode, we expect to be prompted to input this value.

    p = Parent()
    p_conf = {"a": 10, "b": "foo"}

    c_value = [3.14, 2.7]

    with patch("zookeeper.utils.prompt", return_value=str(c_value)) as prompt:
        p.configure(p_conf, name="parent", interactive=True)

    assert p.child.grand_child.c == c_value
    prompt.assert_called_once()


def test_configure_interactive_prompt_for_subcomponent_choice():
    # Configure a parent with an unspecified child subcomponent. In interactive
    # mode, we expect to be prompted to choose from the list of defined,
    # concrete sub-components.

    class A(Component):
        a: int = 5

        def __call__(self):
            return self.a

    class B(A):
        def __call__(self):
            return super().__call__() ** 3

    class B2(B):
        def __call__(self):
            return super().__call__() + 1

    class C(A):
        def __call__(self):
            return super().__call__() * 2

    class Parent(Component):
        child: A

        def __call__(self):
            return self.child.__call__()

    p = Parent()
    p_conf = {}

    # The prompt lists the concrete subclasses (alphabetically) and asks for an
    # an integer input corresponding to an index in this list. The response '3'
    # therefore selects `B2`.
    with patch("zookeeper.utils.prompt", return_value=str(3)) as prompt:
        p.configure(p_conf, interactive=True)

    assert isinstance(p.child, B2)
    assert p() == 126
    prompt.assert_called_once()


def test_str_and_repr(Parent):
    # `__str__` and `__repr__` should give formatted strings that represent
    # nested components nicely.

    p = Parent()
    p_conf = {"a": 10, "b": "foo", "c": [1.5, -1.2]}

    p.configure(p_conf, name="parent")

    assert (
        unstyle(repr(p))
        == """Parent(a=10, b="foo", child=Child(b="foo", grand_child=GrandChild1(a=10, b="foo", c=[1.5, -1.2])))"""
    )
    assert (
        unstyle(str(p))
        == """Parent(
    a = 10,
    b = "foo",
    child = Child(
        b = "foo",
        grand_child = GrandChild1(
            a = 10,
            b = "foo",
            c = [1.5, -1.2]
        )
    )
)"""
    )
