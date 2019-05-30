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

    return Hyper()


def test_defaults(hyper):
    assert hyper.foo == [1, 2, 3]
    assert hyper.bar == 0.5
    assert hyper.barx2 == 1.0
    assert hyper.baz == "string"


def test_parse(hyper):
    hyper.parse("foo=[4, 5, 6],bar=1.,baz='changed'")
    assert hyper.foo == [4, 5, 6]
    assert hyper.bar == 1.0
    assert hyper.barx2 == 2.0
    assert hyper.baz == "changed"


def test_parse_fail(hyper):
    with pytest.raises(ValueError):
        hyper.parse("foo=[4, 5, 6")
    with pytest.raises(ValueError):
        hyper.parse("unknown=3")


def test_immutability(hyper):
    with pytest.raises(AttributeError):
        hyper.new_prop = 3


def test_get_private_methods(hyper):
    with pytest.raises(KeyError):
        hyper["__dict__"]


def test_repr(hyper):
    output = """Hyper(
    bar=0.5,
    barx2=1.0,
    baz=string,
    foo=[1, 2, 3]
)"""
    assert click.unstyle(str(hyper)) == output
    assert click.unstyle(repr(hyper)) == output
