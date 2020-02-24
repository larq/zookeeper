from pathlib import Path

import numpy as np
import pytest
import tensorflow_datasets as tfds

from zookeeper import Field, component
from zookeeper.core.component import configure
from zookeeper.tf import DummyData


@component
class TinyVOC(DummyData):
    num_classes: int = Field(20)
    size: int = Field(32)
    filename: Path = Field(
        lambda: Path(__file__).parent / "fixtures" / "dummy_VOC2007.npy"
    )
    info: tfds.core.DatasetInfo = Field(lambda: tfds.builder("voc/2007").info)


def convert_to_numpy(dataset):
    output = []
    for sample in tfds.as_numpy(dataset):
        output.append(sample)

    return np.array(output)


def assert_dict_equal(dict1, dict2):
    assert dict1.keys() == dict2.keys()

    for key in dict1.keys():
        if isinstance(dict1[key], dict):
            assert_dict_equal(dict1[key], dict2[key])
        else:
            assert np.all(dict1[key] == dict2[key])


@pytest.mark.parametrize("dataset", [TinyVOC()])
def test_dummy_data(dataset):
    configure(dataset, {})  # To ensure __post_configure__ is called
    assert dataset.num_examples("train") == dataset.size
    assert dataset.num_examples("validation") == dataset.size

    loaded_data = np.load(dataset.filename, allow_pickle=True)

    # Compare loaded TF data to numpy loaded data
    train_data, num_examples = dataset.train()
    train_data = convert_to_numpy(train_data)
    assert len(train_data) == len(loaded_data) == num_examples == dataset.size
    for loaded, train in zip(loaded_data, train_data):
        assert_dict_equal(loaded, train)

    validation_data, num_examples = dataset.validation()
    validation_data = convert_to_numpy(validation_data)
    assert len(validation_data) == len(loaded_data) == num_examples == dataset.size
    for loaded, validation in zip(loaded_data, validation_data):
        assert_dict_equal(loaded, validation)
