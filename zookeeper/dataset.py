from typing import Optional

import tensorflow as tf
import tensorflow_datasets as tfds
from zookeeper import Component


def base_splits(split):
    """
    Splits can be merged, e.g. `tfds.Split.TRAIN + tfds.Split.Validation`. For
    such composite splits, find and return a list of 'base splits'.
    """

    if isinstance(split, tfds.core.splits._SplitMerged):
        return base_splits(split._split1) + base_splits(split._split2)
    return [split]


class Dataset(Component):
    """
    A wrapper around a TensorFlowDatasets dataset.
    """

    # The TensorFlowDatasets name.
    name: str

    # The TensorFlowDatasets version.
    version: Optional[str] = None

    # The directory that the dataset is stored in.
    data_dir: Optional[str] = None

    # Train and validation splits. A validation split is not required.
    train_split: str
    validation_split: Optional[str] = None

    def validate_configuration(self):
        super().validate_configuration()

        # Check that the name corresponds to a valid TensorFlow dataset.
        builder_names = tfds.list_builders()
        if self.name not in builder_names:
            raise ValueError(
                f"'{self.__component_name__}.name' has invalid value '{self.name}'. "
                "Valid values:\n    " + ",\n    ".join(builder_names)
            )

        # Check that the `train_split` is valid.
        if self.train_split is None or any(
            s not in self.splits for s in base_splits(self.train_split)
        ):
            raise ValueError(
                f"'{self.__component_name__}.train_split' has invalid value "
                f"'{self.train_split}'. Valid values:\n    "
                + ",\n    ".join(self.splits.keys())
            )

        # Check that `validation_split` is valid (`None` is allowed).
        if self.validation_split is not None and any(
            s not in self.splits for s in base_splits(self.train_split)
        ):
            raise ValueError(
                f"'{self.__component_name__}.train_split' has invalid value "
                f"'{self.train_split}'. Valid values:\n    "
                + ",\n    ".join([None] + self.splits.keys())
            )

    @property
    def name_version(self):
        return f"{self.name}:{self.version}" if self.version else self.name

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
            if "labels" in features:
                return features["labels"].feature.num_classes
            if "objects" in features and "label" in features["objects"]:
                return features["objects"]["label"].num_classes
        except Exception:
            pass
        raise ValueError("Unable to determine the number of classes.")

    def num_examples(self, split) -> int:
        """Compute the number of examples in a given split."""

        return sum(self.splits[s].num_examples for s in base_splits(split))

    def load(self, split, decoders, shuffle) -> tf.data.Dataset:
        """Return a `tf.data.Dataset` object representing the requested split."""

        return tfds.load(
            name=self.name_version,
            split=split,
            data_dir=self.data_dir,
            decoders=decoders,
            as_dataset_kwargs={"shuffle_files": shuffle},
        )

    def train_data(self, decoders=None) -> tf.data.Dataset:
        return self.load(self.train_split, decoders=decoders, shuffle=True)

    @property
    def num_train_examples(self) -> int:
        return self.num_examples(self.train_split)

    def validation_data(self, decoders=None) -> tf.data.Dataset:
        if self.validation_split is None:
            raise ValueError(
                f"Dataset {self.__class__.__name__} is not configured with a "
                "validation split."
            )
        return self.load(self.validation_split, decoders=decoders, shuffle=False)

    @property
    def num_validation_examples(self) -> int:
        if self.validation_split is None:
            raise ValueError(
                f"Dataset {self.__class__.__name__} is not configured with a "
                "validation split."
            )
        return self.num_examples(self.validation_split)
