# Forked from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/training/python/training/hparam_test.py
"""Tests for hparam."""
import pytest
import click
from zookeeper.hparam import HParams


class Hyper(HParams):
    foo = [1, 2, 3]
    bar = 0.5
    baz = "string"

    @property
    def barx2(self):
        return self.bar * 2

    def bar_func(self):
        return self.bar


@pytest.fixture
def hyper():
    return Hyper()


@pytest.fixture
def hyper_with_nested():
    class Child(HParams):
        c = -1.5
        d = "aeiou"

    class Parent(HParams):
        a = 4.9
        b = "some string"
        child = Child()

    return Parent()


def test_defaults(hyper):
    assert hyper.foo == [1, 2, 3]
    assert hyper.bar == 0.5
    assert hyper.barx2 == 1.0
    assert hyper.baz == "string"
    assert hyper.bar_func() == hyper.bar


def test_parse(hyper):
    hyper.parse("foo=[4, 5, 6],bar=1.,baz='changed'")
    assert hyper.foo == [4, 5, 6]
    assert hyper.bar == 1.0
    assert hyper.barx2 == 2.0
    assert hyper.baz == "changed"
    assert hyper.bar_func() == hyper.bar


def test_spaced_parse(hyper):
    hyper.parse("foo=[4, 5, 6], bar=1.,baz='spaced argument'")
    assert hyper.foo == [4, 5, 6]
    assert hyper.bar == 1.0
    assert hyper.barx2 == 2.0
    assert hyper.baz == "spaced argument"
    assert hyper.bar_func() == hyper.bar


def test_parse_fail(hyper):
    with pytest.raises(ValueError):
        hyper.parse("foo=[4, 5, 6")
    with pytest.raises(ValueError):
        hyper.parse("unknown=3")


def test_immutability(hyper):
    with pytest.raises(AttributeError):
        hyper.new_prop = 3


def test_key_error(hyper):
    with pytest.raises(KeyError):
        hyper["__dict__"]
    with pytest.raises(KeyError):
        hyper["unknown"]


def test_repr(hyper):
    output = "Hyper(bar=0.5,bar_func=<callable>,barx2=1.0,baz=string,foo=[1, 2, 3])"
    assert repr(hyper) == output


def test_repr_nested(hyper_with_nested):
    output = "Parent(a=4.9,b=some string,child=Child(c=-1.5,d=aeiou))"
    assert repr(hyper_with_nested) == output


def test_str(hyper):
    output = """Hyper(
    bar=0.5,
    bar_func=<callable>,
    barx2=1.0,
    baz=string,
    foo=[1, 2, 3]
)"""
    assert click.unstyle(str(hyper)) == output


def test_str_nested(hyper_with_nested):
    output = """Parent(
    a=4.9,
    b=some string,
    child=Child(
        c=-1.5,
        d=aeiou
    )
)"""
    assert click.unstyle(str(hyper_with_nested)) == output


def test_init_kwargs(hyper):
    new_hyper = Hyper(foo="updated_foo", bar=-100, new_name="new_value")
    # Updated
    assert new_hyper.foo == "updated_foo"
    assert new_hyper.bar == -100
    assert new_hyper.barx2 == -200
    # Added
    assert new_hyper.new_name == "new_value"
    # The same
    assert new_hyper.baz == hyper.baz
    # Invalid keys
    with pytest.raises(ValueError):
        new_hyper = Hyper(_new_name="new_value")
    with pytest.raises(ValueError):
        new_hyper = Hyper(parse=lambda x: x ** 2)
