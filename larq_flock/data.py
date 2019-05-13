import functools
import inspect
import tensorflow as tf
import tensorflow_datasets as tfds


class Dataset:
    def __init__(self, dataset_name, preprocess_fn, use_val_split=False, data_dir=None):
        self.dataset_name = dataset_name
        self.preprocess_fn = preprocess_fn
        self.data_dir = data_dir

        dataset_builder = tfds.builder(dataset_name)
        info = dataset_builder.info
        splits = dataset_builder.info.splits
        if tfds.Split.TRAIN not in splits:
            raise ValueError("To train we require a train split in the dataset.")
        if tfds.Split.TEST not in splits and tfds.Split.VALIDATION not in splits:
            raise ValueError("We require a test or validation split in the dataset.")
        if (
            not info.supervised_keys
            or "image" not in info.supervised_keys
            or "label" not in info.supervised_keys
        ):
            raise NotImplementedError(
                "We currently only support supervised image classification"
            )
        self.num_classes = info.features["label"].num_classes
        self.input_shape = getattr(
            preprocess_fn, "input_shape", info.features["image"].shape
        )

        self.train_split = tfds.Split.TRAIN
        self.train_examples = splits[self.train_split].num_examples
        if tfds.Split.TEST in splits:
            self.test_split = tfds.Split.TEST
            self.test_examples = splits[self.test_split].num_examples
        if tfds.Split.VALIDATION in splits:
            self.validation_split = tfds.Split.VALIDATION
            self.validation_examples = splits[self.validation_split].num_examples
        else:
            if use_val_split == True:
                self.train_split = tfds.Split.TRAIN.subsplit(tfds.percent[:90])
                all_train_examples = splits[tfds.Split.TRAIN].num_examples
                self.train_examples = int(all_train_examples * 0.9)
                self.validation_split = tfds.Split.TRAIN.subsplit(tfds.percent[-10:])
                self.validation_examples = all_train_examples - self.train_examples
            else:
                self.validation_split = self.test_split
                self.validation_examples = self.test_examples

    def _get_dataset(self, split):
        return tfds.load(name=self.dataset_name, split=split, data_dir=self.data_dir)

    def _map_fn(self, data, training=False):
        if "training" in inspect.getfullargspec(self.preprocess_fn).args:
            image = self.preprocess_fn(data["image"], training=training)
        else:
            image = self.preprocess_fn(data["image"])
        label = tf.one_hot(data["label"], self.num_classes)
        return image, label

    def train_data(self, batch_size):
        return (
            self._get_dataset(self.train_split)
            .shuffle(10 * batch_size)
            .repeat()
            .map(
                functools.partial(self._map_fn, training=True),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    def validation_data(self, batch_size):
        return self._get_eval_data(self._get_dataset(self.validation_split), batch_size)

    def test_data(self, batch_size):
        return self._get_eval_data(self._get_dataset(self.test_split), batch_size)

    def _get_eval_data(self, dataset, batch_size):
        return (
            dataset.repeat()
            .map(self._map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
