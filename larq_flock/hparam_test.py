# Forked from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/training/python/training/hparam_test.py
"""Tests for hparam."""
import pytest
from larq_flock import hparam


def test_empty():
    hparams = hparam.HParams()
    assert {} == hparams.values()
    hparams.parse("")
    assert {} == hparams.values()
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        hparams.parse("xyz=123")


def test_contains():
    hparams = hparam.HParams(foo=1)
    assert "foo" in hparams
    assert not ("bar" in hparams)


def test_some_values():
    hparams = hparam.HParams(aaa=1, b=2.0, c_c="relu6", d="/a/b=c/d")
    assert {"aaa": 1, "b": 2.0, "c_c": "relu6", "d": "/a/b=c/d"} == hparams.values()
    expected_str = "[('aaa', 1), ('b', 2.0), ('c_c', 'relu6'), " "('d', '/a/b=c/d')]"
    assert expected_str == str(hparams.__str__())
    assert expected_str == str(hparams)
    assert 1 == hparams.aaa
    assert 2.0 == hparams.b
    assert "relu6" == hparams.c_c
    assert "/a/b=c/d" == hparams.d
    hparams.parse("aaa=12")
    assert {"aaa": 12, "b": 2.0, "c_c": "relu6", "d": "/a/b=c/d"} == hparams.values()
    assert 12 == hparams.aaa
    assert 2.0 == hparams.b
    assert "relu6" == hparams.c_c
    assert "/a/b=c/d" == hparams.d
    hparams.parse("c_c=relu4, b=-2.0e10")
    assert {
        "aaa": 12,
        "b": -2.0e10,
        "c_c": "relu4",
        "d": "/a/b=c/d",
    } == hparams.values()
    assert 12 == hparams.aaa
    assert -2.0e10 == hparams.b
    assert "relu4" == hparams.c_c
    assert "/a/b=c/d" == hparams.d
    hparams.parse("c_c=,b=0,")
    assert {"aaa": 12, "b": 0, "c_c": "", "d": "/a/b=c/d"} == hparams.values()
    assert 12 == hparams.aaa
    assert 0.0 == hparams.b
    assert "" == hparams.c_c
    assert "/a/b=c/d" == hparams.d
    hparams.parse('c_c=2.3",b=+2,')
    assert 2.0 == hparams.b
    assert '2.3"' == hparams.c_c
    hparams.parse("d=/a/b/c/d,aaa=11,")
    assert 11 == hparams.aaa
    assert 2.0 == hparams.b
    assert '2.3"' == hparams.c_c
    assert "/a/b/c/d" == hparams.d
    hparams.parse("b=1.5,d=/a=b/c/d,aaa=10,")
    assert 10 == hparams.aaa
    assert 1.5 == hparams.b
    assert '2.3"' == hparams.c_c
    assert "/a=b/c/d" == hparams.d
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        hparams.parse("x=123")
    with pytest.raises(ValueError, match="Could not parse"):
        hparams.parse("aaa=poipoi")
    with pytest.raises(ValueError, match="Could not parse"):
        hparams.parse("aaa=1.0")
    with pytest.raises(ValueError, match="Could not parse"):
        hparams.parse("b=12x")
    with pytest.raises(ValueError, match="Could not parse"):
        hparams.parse("b=relu")
    with pytest.raises(ValueError, match="Must not pass a list"):
        hparams.parse("aaa=[123]")
    assert 10 == hparams.aaa
    assert 1.5 == hparams.b
    assert '2.3"' == hparams.c_c
    assert "/a=b/c/d" == hparams.d


def testWith_period_in_variable_name():
    hparams = hparam.HParams()
    hparams.add_hparam(name="a.b", value=0.0)
    hparams.parse("a.b=1.0")
    assert 1.0 == getattr(hparams, "a.b")
    hparams.add_hparam(name="c.d", value=0.0)
    with pytest.raises(ValueError, match="Could not parse"):
        hparams.parse("c.d=abc")
    hparams.add_hparam(name="e.f", value="")
    hparams.parse("e.f=abc")
    assert "abc" == getattr(hparams, "e.f")
    hparams.add_hparam(name="d..", value=0.0)
    hparams.parse("d..=10.0")
    assert 10.0 == getattr(hparams, "d..")


def test_set_from_map():
    hparams = hparam.HParams(a=1, b=2.0, c="tanh")
    hparams.override_from_dict({"a": -2, "c": "identity"})
    assert {"a": -2, "c": "identity", "b": 2.0} == hparams.values()

    hparams = hparam.HParams(x=1, b=2.0, d=[0.5])
    hparams.override_from_dict({"d": [0.1, 0.2, 0.3]})
    assert {"d": [0.1, 0.2, 0.3], "x": 1, "b": 2.0} == hparams.values()


def test_function():
    def f(x):
        return x

    hparams = hparam.HParams(function=f)
    assert hparams.function == f

    json_str = hparams.to_json()
    assert json_str == "{}"


def test_bool_parsing():
    for value in "true", "false", "True", "False", "1", "0":
        for initial in False, True:
            hparams = hparam.HParams(use_gpu=initial)
            hparams.parse("use_gpu=" + value)
            assert hparams.use_gpu == (value in ["True", "true", "1"])


def test_bool_parsing_fail():
    hparams = hparam.HParams(use_gpu=True)
    with pytest.raises(ValueError, match=r"Could not parse.*use_gpu"):
        hparams.parse("use_gpu=yep")


def test_lists():
    hparams = hparam.HParams(aaa=[1], b=[2.0, 3.0], c_c=["relu6"])
    assert {"aaa": [1], "b": [2.0, 3.0], "c_c": ["relu6"]} == hparams.values()
    assert [1] == hparams.aaa
    assert [2.0, 3.0] == hparams.b
    assert ["relu6"] == hparams.c_c
    hparams.parse("aaa=[12]")
    assert [12] == hparams.aaa
    hparams.parse("aaa=[12,34,56]")
    assert [12, 34, 56] == hparams.aaa
    hparams.parse("c_c=[relu4,relu12],b=[1.0]")
    assert ["relu4", "relu12"] == hparams.c_c
    assert [1.0] == hparams.b
    hparams.parse("c_c=[],aaa=[-34]")
    assert [-34] == hparams.aaa
    assert [] == hparams.c_c
    hparams.parse("c_c=[_12,3'4\"],aaa=[+3]")
    assert [3] == hparams.aaa
    assert ["_12", "3'4\""] == hparams.c_c
    with pytest.raises(ValueError, match="Unknown hyperparameter"):
        hparams.parse("x=[123]")
    with pytest.raises(ValueError, match="Could not parse"):
        hparams.parse("aaa=[poipoi]")
    with pytest.raises(ValueError, match="Could not parse"):
        hparams.parse("aaa=[1.0]")
    with pytest.raises(ValueError, match="Could not parse"):
        hparams.parse("b=[12x]")
    with pytest.raises(ValueError, match="Could not parse"):
        hparams.parse("b=[relu]")
    with pytest.raises(ValueError, match="Must pass a list"):
        hparams.parse("aaa=123")


def test_parse_values_with_index_assigment1():
    """Assignment to an index position."""
    parse_dict = hparam.parse_values("arr[1]=10", {"arr": int})
    assert len(parse_dict) == 1
    assert isinstance(parse_dict["arr"], dict)
    assert parse_dict["arr"] == {1: 10}


def test_parse_values_with_index_assigment1_ignore_unknown():
    """Assignment to an index position."""
    parse_dict = hparam.parse_values("arr[1]=10,b=5", {"arr": int}, ignore_unknown=True)
    assert len(parse_dict) == 1
    assert isinstance(parse_dict["arr"], dict)
    assert parse_dict["arr"] == {1: 10}


def test_parse_values_with_index_assigment2():
    """Assignment to multiple index positions."""
    parse_dict = hparam.parse_values("arr[0]=10,arr[5]=20", {"arr": int})
    assert len(parse_dict) == 1
    assert isinstance(parse_dict["arr"], dict)
    assert parse_dict["arr"] == {0: 10, 5: 20}


def test_parse_values_with_index_assigment2_ignore_unknown():
    """Assignment to multiple index positions."""
    parse_dict = hparam.parse_values(
        "arr[0]=10,arr[5]=20,foo=bar", {"arr": int}, ignore_unknown=True
    )
    assert len(parse_dict) == 1
    assert isinstance(parse_dict["arr"], dict)
    assert parse_dict["arr"] == {0: 10, 5: 20}


def test_parse_values_with_index_assigment3():
    """Assignment to index positions in multiple names."""
    parse_dict = hparam.parse_values(
        "arr[0]=10,arr[1]=20,L[5]=100,L[10]=200", {"arr": int, "L": int}
    )
    assert len(parse_dict) == 2
    assert isinstance(parse_dict["arr"], dict)
    assert parse_dict["arr"] == {0: 10, 1: 20}
    assert isinstance(parse_dict["L"], dict)
    assert parse_dict["L"] == {5: 100, 10: 200}


def test_parse_values_with_index_assigment3_ignore_unknown():
    """Assignment to index positions in multiple names."""
    parse_dict = hparam.parse_values(
        "arr[0]=10,C=5,arr[1]=20,B[0]=kkk,L[5]=100,L[10]=200",
        {"arr": int, "L": int},
        ignore_unknown=True,
    )
    assert len(parse_dict) == 2
    assert isinstance(parse_dict["arr"], dict)
    assert parse_dict["arr"] == {0: 10, 1: 20}
    assert isinstance(parse_dict["L"], dict)
    assert parse_dict["L"] == {5: 100, 10: 200}


def test_parse_values_with_index_assigment4():
    """Assignment of index positions and scalars."""
    parse_dict = hparam.parse_values(
        "x=10,arr[1]=20,y=30", {"x": int, "y": int, "arr": int}
    )
    assert len(parse_dict) == 3
    assert isinstance(parse_dict["arr"], dict)
    assert parse_dict["arr"] == {1: 20}
    assert parse_dict["x"] == 10
    assert parse_dict["y"] == 30


def test_parse_values_with_index_assigment4_ignore_unknown():
    """Assignment of index positions and scalars."""
    parse_dict = hparam.parse_values(
        "x=10,foo[0]=bar,arr[1]=20,zzz=78,y=30",
        {"x": int, "y": int, "arr": int},
        ignore_unknown=True,
    )
    assert len(parse_dict) == 3
    assert isinstance(parse_dict["arr"], dict)
    assert parse_dict["arr"] == {1: 20}
    assert parse_dict["x"] == 10
    assert parse_dict["y"] == 30


def test_parse_values_with_index_assigment5():
    """Different variable types."""
    parse_dict = hparam.parse_values(
        "a[0]=5,b[1]=true,c[2]=abc,d[3]=3.14",
        {"a": int, "b": bool, "c": str, "d": float},
    )
    assert set(parse_dict.keys()) == {"a", "b", "c", "d"}
    assert isinstance(parse_dict["a"], dict)
    assert parse_dict["a"] == {0: 5}
    assert isinstance(parse_dict["b"], dict)
    assert parse_dict["b"] == {1: True}
    assert isinstance(parse_dict["c"], dict)
    assert parse_dict["c"] == {2: "abc"}
    assert isinstance(parse_dict["d"], dict)
    assert parse_dict["d"] == {3: 3.14}


def test_parse_values_with_index_assigment5_ignore_unknown():
    """Different variable types."""
    parse_dict = hparam.parse_values(
        "a[0]=5,cc=4,b[1]=true,c[2]=abc,mm=2,d[3]=3.14",
        {"a": int, "b": bool, "c": str, "d": float},
        ignore_unknown=True,
    )
    assert set(parse_dict.keys()) == {"a", "b", "c", "d"}
    assert isinstance(parse_dict["a"], dict)
    assert parse_dict["a"] == {0: 5}
    assert isinstance(parse_dict["b"], dict)
    assert parse_dict["b"] == {1: True}
    assert isinstance(parse_dict["c"], dict)
    assert parse_dict["c"] == {2: "abc"}
    assert isinstance(parse_dict["d"], dict)
    assert parse_dict["d"] == {3: 3.14}


def test_parse_values_with_bad_index_assigment1():
    """Reject assignment of list to variable type."""
    with pytest.raises(ValueError, match=r"Assignment of a list to a list index."):
        hparam.parse_values("arr[1]=[1,2,3]", {"arr": int})


def test_parse_values_with_bad_index_assigment1_ignore_unknown():
    """Reject assignment of list to variable type."""
    with pytest.raises(ValueError, match=r"Assignment of a list to a list index."):
        hparam.parse_values("arr[1]=[1,2,3],c=8", {"arr": int}, ignore_unknown=True)


def test_parse_values_with_bad_index_assigment2():
    """Reject if type missing."""
    with pytest.raises(ValueError, match=r"Unknown hyperparameter type for arr"):
        hparam.parse_values("arr[1]=5", {})


def test_parse_values_with_bad_index_assigment2_ignore_unknown():
    """Ignore missing type."""
    hparam.parse_values("arr[1]=5", {}, ignore_unknown=True)


def test_parse_values_with_bad_index_assigment3():
    """Reject type of the form name[index]."""
    with pytest.raises(ValueError, match="Unknown hyperparameter type for arr"):
        hparam.parse_values("arr[1]=1", {"arr[1]": int})


def test_parse_values_with_bad_index_assigment3_ignore_unknown():
    """Ignore type of the form name[index]."""
    hparam.parse_values("arr[1]=1", {"arr[1]": int}, ignore_unknown=True)


def test_with_reused_variable():
    with pytest.raises(ValueError, match="Multiple assignments to variable 'x'"):
        hparam.parse_values("x=1,x=1", {"x": int})

    with pytest.raises(ValueError, match="Multiple assignments to variable 'arr'"):
        hparam.parse_values("arr=[100,200],arr[0]=10", {"arr": int})

    with pytest.raises(
        ValueError, match=r"Multiple assignments to variable \'arr\[0\]\'"
    ):
        hparam.parse_values("arr[0]=10,arr[0]=20", {"arr": int})

    with pytest.raises(ValueError, match="Multiple assignments to variable 'arr'"):
        hparam.parse_values("arr[0]=10,arr=[100]", {"arr": int})


def test_json():
    hparams = hparam.HParams(aaa=1, b=2.0, c_c="relu6", d=True)
    assert {"aaa": 1, "b": 2.0, "c_c": "relu6", "d": True} == hparams.values()
    assert 1 == hparams.aaa
    assert 2.0 == hparams.b
    assert "relu6" == hparams.c_c
    hparams.parse_json('{"aaa": 12, "b": 3.0, "c_c": "relu4", "d": false}')
    assert {"aaa": 12, "b": 3.0, "c_c": "relu4", "d": False} == hparams.values()
    assert 12 == hparams.aaa
    assert 3.0 == hparams.b
    assert "relu4" == hparams.c_c

    json_str = hparams.to_json()
    hparams2 = hparam.HParams(aaa=10, b=20.0, c_c="hello", d=False)
    hparams2.parse_json(json_str)
    assert 12 == hparams2.aaa
    assert 3.0 == hparams2.b
    assert "relu4" == hparams2.c_c
    assert False == hparams2.d

    hparams3 = hparam.HParams(aaa=123)
    assert '{"aaa": 123}' == hparams3.to_json()
    assert '{\n  "aaa": 123\n}' == hparams3.to_json(indent=2)
    assert '{"aaa"=123}' == hparams3.to_json(separators=(";", "="))

    hparams4 = hparam.HParams(aaa=123, b="hello", c_c=False)
    assert '{"aaa": 123, "b": "hello", "c_c": false}' == hparams4.to_json(
        sort_keys=True
    )


def test_set_hparam():
    hparams = hparam.HParams(aaa=1, b=2.0, c_c="relu6", d=True)
    assert {"aaa": 1, "b": 2.0, "c_c": "relu6", "d": True} == hparams.values()
    assert 1 == hparams.aaa
    assert 2.0 == hparams.b
    assert "relu6" == hparams.c_c

    hparams.set_hparam("aaa", 12)
    hparams.set_hparam("b", 3.0)
    hparams.set_hparam("c_c", "relu4")
    hparams.set_hparam("d", False)
    assert {"aaa": 12, "b": 3.0, "c_c": "relu4", "d": False} == hparams.values()
    assert 12 == hparams.aaa
    assert 3.0 == hparams.b
    assert "relu4" == hparams.c_c


def test_set_hparam_list_non_list_mismatch():
    hparams = hparam.HParams(a=1, b=[2.0, 3.0])
    with pytest.raises(ValueError, match=r"Must not pass a list"):
        hparams.set_hparam("a", [1.0])
    with pytest.raises(ValueError, match=r"Must pass a list"):
        hparams.set_hparam("b", 1.0)


def test_set_hparam_type_mismatch():
    hparams = hparam.HParams(
        int_=1, str_="str", bool_=True, float_=1.1, list_int=[1, 2], none=None
    )

    with pytest.raises(ValueError):
        hparams.set_hparam("str_", 2.2)
    with pytest.raises(ValueError):
        hparams.set_hparam("int_", False)
    with pytest.raises(ValueError):
        hparams.set_hparam("bool_", 1)
    with pytest.raises(ValueError):
        hparams.set_hparam("int_", 2.2)
    with pytest.raises(ValueError):
        hparams.set_hparam("list_int", [2, 3.3])
    with pytest.raises(ValueError):
        hparams.set_hparam("int_", "2")

    # Casting int to float is OK
    hparams.set_hparam("float_", 1)

    # Getting stuck with NoneType :(
    hparams.set_hparam("none", "1")
    assert "1" == hparams.none


def test_get():
    hparams = hparam.HParams(aaa=1, b=2.0, c_c="relu6", d=True, e=[5.0, 6.0])

    # Existing parameters with default=None.
    assert 1 == hparams.get("aaa")
    assert 2.0 == hparams.get("b")
    assert "relu6" == hparams.get("c_c")
    assert True == hparams.get("d")
    assert [5.0, 6.0] == hparams.get("e", None)

    # Existing parameters with compatible defaults.
    assert 1 == hparams.get("aaa", 2)
    assert 2.0 == hparams.get("b", 3.0)
    assert 2.0 == hparams.get("b", 3)
    assert "relu6" == hparams.get("c_c", "default")
    assert True == hparams.get("d", True)
    assert [5.0, 6.0] == hparams.get("e", [1.0, 2.0, 3.0])
    assert [5.0, 6.0] == hparams.get("e", [1, 2, 3])

    # Existing parameters with incompatible defaults.
    with pytest.raises(ValueError):
        hparams.get("aaa", 2.0)
    with pytest.raises(ValueError):
        hparams.get("b", False)
    with pytest.raises(ValueError):
        hparams.get("c_c", [1, 2, 3])
    with pytest.raises(ValueError):
        hparams.get("d", "relu")
    with pytest.raises(ValueError):
        hparams.get("e", 123.0)
    with pytest.raises(ValueError):
        hparams.get("e", ["a", "b", "c"])

    # Nonexistent parameters.
    assert None == hparams.get("unknown")
    assert 123 == hparams.get("unknown", 123)
    assert [1, 2, 3] == hparams.get("unknown", [1, 2, 3])


def test_del():
    hparams = hparam.HParams(aaa=1, b=2.0)

    with pytest.raises(ValueError):
        hparams.set_hparam("aaa", "will fail")
    with pytest.raises(ValueError):
        hparams.add_hparam("aaa", "will fail")

    hparams.del_hparam("aaa")
    hparams.add_hparam("aaa", "will work")
    assert "will work" == hparams.get("aaa")

    hparams.set_hparam("aaa", "still works")
    assert "still works" == hparams.get("aaa")
