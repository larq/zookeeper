import pytest
import mock
from larq_flock import utils


@mock.patch("os.makedirs")
def test_cache_dir(os_makedirs):
    assert utils.get_cache_dir(None, "mnist") == None
    assert utils.get_cache_dir("memory", "mnist") == ""
    with mock.patch("glob.glob", return_value=[]):
        assert utils.get_cache_dir("foo", "mnist") == "foo/mnist"
    with mock.patch("glob.glob", side_effect=[["bar"], ["baz"], []]):
        assert utils.get_cache_dir("foo", "mnist") == "foo/mnist_2"
    with mock.patch("glob.glob", return_value=["bar"]):
        with pytest.raises(RuntimeError):
            utils.get_cache_dir("foo", "mnist")
