import pytest
from unittest import mock
from zookeeper import data, Preprocessing


class MockPreprocessing(Preprocessing):
    def inputs(self, data):
        return data["image"]

    def outputs(self, data):
        return data["label"]


@mock.patch("os.makedirs")
def test_cache_dir(os_makedirs):
    dataset = data.Dataset("mnist", MockPreprocessing)
    assert dataset.get_cache_path("train") == None

    dataset.cache_dir = "memory"
    assert dataset.get_cache_path("train") == ""

    dataset.cache_dir = "foo"
    with mock.patch("glob.glob", return_value=[]):
        assert dataset.get_cache_path("train") == "foo/mnist/train"
    with mock.patch("glob.glob", side_effect=[["bar"], ["baz"], []]):
        assert dataset.get_cache_path("train") == "foo/mnist_2/train"
    with mock.patch("glob.glob", return_value=["bar"]):
        with pytest.raises(RuntimeError):
            dataset.get_cache_path("train")
