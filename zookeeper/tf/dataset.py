import abc
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from zookeeper.core.field import Field


class Dataset(abc.ABC):
    """
    An abstract base class to encapsulate a dataset. Concrete sub-classes must
    implement the `train` method, and optionally the `validation` method.
    """

    @abc.abstractmethod
    def train(self, decoders=None) -> Tuple[tf.data.Dataset, int]:
        """
        Return a tuple of the training dataset and the number of training
        examples in the dataset.
        """

        raise NotImplementedError

    def validation(self, decoders=None) -> Tuple[tf.data.Dataset, int]:
        """
        Return a tuple of the validation dataset and the number of validation
        examples in the dataset. By default, raises an error that no validation
        data is provided.
        """

        raise ValueError(
            f"Dataset '{self.__class__.__name__}' is not configured with validation "
            "data."
        )


def base_splits(split):
    """
    Splits can be merged, e.g. `tfds.Split.TRAIN + tfds.Split.Validation` or
    `"train+validation"`. For such composite splits, find and return a list of
    'base splits'.
    """

    if "+" in split:
        return split.split("+")
    elif isinstance(split, tfds.core.splits._SplitMerged):
        return base_splits(split._split1) + base_splits(split._split2)
    return [split]


class TFDSDataset(Dataset):
    """
    A wrapper around a TensorFlowDatasets dataset.
    """

    # The TensorFlowDatasets name, which may specify a builder config and/or
    # version, e.g. "imagenet2012:4.0.0"
    name: str = Field()

    # The directory that the dataset is stored in.
    data_dir: Optional[str] = Field(None)

    # Train and validation splits. A validation split is not required.
    train_split: str = Field()
    validation_split: Optional[str] = Field(None)

    @property
    def info(self):
        if not hasattr(self, "_info"):
            self._info = tfds.builder(self.name, data_dir=self.data_dir).info
        return self._info

    @property
    def splits(self):
        return self.info.splits

    @property
    def num_classes(self) -> int:
        try:
            features = self.info.features
            if "label" in features:
                return features["label"].num_classes
            if "labels" in features and hasattr(features["labels"], "feature"):
                return features["labels"].feature.num_classes
            if "objects" in features and "label" in features["objects"]:
                return features["objects"]["label"].num_classes
        except Exception:
            pass
        raise ValueError("Unable to determine the number of classes automatically.")

    def num_examples(self, split) -> int:
        """Compute the number of examples in a given split."""

        return sum(self.splits[s].num_examples for s in base_splits(split))

    def load(self, split, decoders, shuffle) -> tf.data.Dataset:
        """Return a `tf.data.Dataset` object representing the requested split."""

        return tfds.load(
            name=self.name,
            split=split,
            data_dir=self.data_dir,
            decoders=decoders,
            as_dataset_kwargs={"shuffle_files": shuffle},
        )

    def train(self, decoders=None) -> Tuple[tf.data.Dataset, int]:
        return (
            self.load(self.train_split, decoders=decoders, shuffle=True),
            self.num_examples(self.train_split),
        )

    def validation(self, decoders=None) -> Tuple[tf.data.Dataset, int]:
        if self.validation_split is None:
            raise ValueError(
                f"Dataset {self.__class__.__name__} is not configured with a "
                "validation split."
            )
        return (
            self.load(self.validation_split, decoders=decoders, shuffle=False),
            self.num_examples(self.validation_split),
        )


class MultiTFDSDataset(Dataset):
    """
    A wrapper around multiple TensorFlowDatasets datasets. This allows a model
    to be trained on data that is combined from multiple datasets.
    """

    # A non-empty mapping from dataset names as keys to splits as values. The
    # training data will be the concatenation of the datasets loaded from each
    # (key, value) pair.
    train_split: Dict[str, str] = Field()

    # As above, a mapping from dataset names as keys to splits as values. May be
    # empty, indicating no validation data.
    validation_split: Dict[str, str] = Field(lambda: {})

    # The directory that the dataset is stored in.
    data_dir: Optional[str] = Field(None)

    def num_examples(self, splits) -> int:
        """
        Compute the total number of examples in the splits specified by the
        dictionary `splits`.
        """

        return sum(
            tfds.builder(name, data_dir=self.data_dir).info.splits[s].num_examples
            for name, split in splits.items()
            for s in base_splits(split)
        )

    def load(self, splits, decoders, shuffle) -> tf.data.Dataset:
        result = None
        for name, split in splits.items():
            dataset = tfds.load(
                name=name,
                split=split,
                data_dir=self.data_dir,
                decoders=decoders,
                as_dataset_kwargs={"shuffle_files": shuffle},
            )
            result = result.concatenate(dataset) if result is not None else dataset
        return result

    def train(self, decoders=None):
        return (
            self.load(self.train_split, decoders=decoders, shuffle=True),
            self.num_examples(self.train_split),
        )

    def validation(self, decoders=None):
        return (
            self.load(self.validation_split, decoders=decoders, shuffle=False),
            self.num_examples(self.validation_split),
        )


class DummyData(TFDSDataset, abc.ABC):
    """
    Abstract class that represents a small subset of a TFDS dataset, loaded from a
    numpy file. Both splits contain the same data.
    """

    # train_split: str = Field("")
    size: int = Field()  # Number of examples in this dummy dataset
    info: tfds.core.DatasetInfo = Field()  # The DatasetInfo of the original dataset
    filename: Path  # Path to the .npy file that stores the data
    train_split: str = Field("")

    def num_examples(self, split: str = "") -> int:
        return self.size

    def __post_configure__(self):
        # Infer output types and shapes from DatasetInfo
        self.output_types = {}
        self.output_shapes = {}
        for key, value in self.info.features.items():
            # If the type is Image, we assume it is stored in encoded form, hence
            # tf.string.
            if type(value) is tfds.features.Image:
                self.output_types[key] = tf.string
                self.output_shapes[key] = None
            else:
                self.output_types[key] = value.dtype
                self.output_shapes[key] = value.shape

    def train(self, decoders=None) -> Tuple[tf.data.Dataset, int]:
        """
        Return a tuple of the training dataset and the number of training
        examples in the dataset.
        """
        dataset = tf.data.Dataset.from_generator(
            self.generate_data,
            output_types=self.output_types,
            output_shapes=self.output_shapes,
        )
        return (dataset, self.num_examples())

    def validation(self, decoders=None) -> Tuple[tf.data.Dataset, int]:
        return self.train()

    def generate_data(self):
        data = np.load(self.filename, allow_pickle=True)

        if len(data) != self.size:
            raise ValueError(
                f"Dataset loaded from {self.filename} contains {len(data)} examples,"
                f"expected {self.size}!"
            )

        for example in data:
            yield example

    @Field
    def name(self):
        return self.__class__.__name__

    @staticmethod
    def create_dummy_data(
        dataset: TFDSDataset,
        num_examples: int,
        decoders: Optional[Dict[str, Union[tfds.decode.Decoder, Dict]]] = {
            "image": tfds.decode.SkipDecoding()
        },
        split: str = "train",
        output_file: Optional[str] = None,
    ) -> None:
        """
        Take an existing TFDSDataset and create a dummy data subset from it, saved to
        the target file. This dummy data can then later be accessed through a custom
        DummyData subclass representing it.
        """

        if split == "train":
            data, num_validation_examples = dataset.train(decoders=decoders)
        elif split == "validation":
            data, num_validation_examples = dataset.validation(decoders=decoders)
        else:
            raise ValueError(
                f"`split` must be either 'train' or 'validation', received '{split}''"
            )

        filename = output_file or f"dummy_{dataset.__class__.__name__}.npy"

        samples = []
        for x in tfds.as_numpy(data):
            if len(samples) == num_examples:
                break

            samples.append(x)

        np.save(filename, samples)
        print(f"Saved dummy data at path {filename}")
