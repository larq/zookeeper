# Forked from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/training/python/training/hparam_test.py
"""Tests for hparam."""
import pytest
import click
from zookeeper.hparam import HParams


@pytest.fixture
def hyper():
    class Hyper(HParams):
        foo = [1, 2, 3]
        bar = 0.5
        baz = "string"

        @property
        def barx2(self):
            return self.bar * 2

        def bar_func(self):
            return self.bar

    return Hyper()


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


def test_str(hyper):
    output = """Hyper(
    bar=0.5,
    bar_func=<callable>,
    barx2=1.0,
    baz=string,
    foo=[1, 2, 3]
)"""
    assert click.unstyle(str(hyper)) == output
