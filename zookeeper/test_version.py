import zookeeper


def test_version():
    assert hasattr(zookeeper, "__version__") and "." in zookeeper.__version__
