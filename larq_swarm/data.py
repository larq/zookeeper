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
            raise ValueError(
                "We currently only support supervised image classification"
            )

        if tfds.Split.VALIDATION not in splits and tfds.Split.TEST in splits:
            self.test_examples = splits[tfds.Split.TEST].num_examples
            if use_val_split == True:
                train_set = tfds.Split.TRAIN.subsplit(tfds.percent[:90])
                val_set = tfds.Split.TRAIN.subsplit(tfds.percent[-10:])
                self.train_examples = int(splits[tfds.Split.TRAIN].num_examples * 0.9)
                self.validation_examples = (
                    splits[tfds.Split.TRAIN].num_examples - self.train_examples
                )
                self._train_dataset, self._validation_dataset, self._test_dataset = tfds.load(
                    name=dataset_name,
                    split=[train_set, val_set, tfds.Split.TEST],
                    data_dir=data_dir,
                )

            else:
                self.train_examples = splits[tfds.Split.TRAIN].num_examples
                self.validation_examples = self.test_examples
                self._train_dataset, self._test_dataset = tfds.load(
                    name=dataset_name,
                    split=[tfds.Split.TRAIN, tfds.Split.TEST],
                    data_dir=data_dir,
                )
                self._validation_dataset = self._test_dataset
        elif tfds.Split.VALIDATION in splits and tfds.Split.TEST not in splits:
            self.train_examples = splits[tfds.Split.TRAIN].num_examples
            self.validation_examples = splits[tfds.Split.VALIDATION].num_examples
            self.test_examples = self.validation_examples
            self._train_dataset, self._validation_dataset = tfds.load(
                name=dataset_name,
                split=[tfds.Split.TRAIN, tfds.Split.VALIDATION],
                data_dir=data_dir,
            )
            self._test_dataset = self._validation_dataset
        else:
            self.train_examples = splits[tfds.Split.TRAIN].num_examples
            self.validation_examples = splits[tfds.Split.VALIDATION].num_examples
            self.test_examples = splits[tfds.Split.TEST].num_examples
            self._train_dataset, self._validation_dataset, self._test_dataset = tfds.load(
                name=dataset_name,
                split=[tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST],
                data_dir=data_dir,
            )

        self.num_classes = info.features["label"].num_classes
        self.image_shape = getattr(
            preprocess_fn, "image_shape", info.features["image"].shape
        )

    def _map_fn(self, data, training=False):
        if "training" in inspect.getfullargspec(self.preprocess_fn).args:
            image = self.preprocess_fn(data["image"], training=training)
        else:
            image = self.preprocess_fn(data["image"])
        label = tf.one_hot(data["label"], self.num_classes)
        return image, label

    def train_data(self, batch_size):
        return (
            self._train_dataset.shuffle(10 * batch_size)
            .repeat()
            .map(
                functools.partial(self._map_fn, training=True),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

    def validation_data(self, batch_size):
        return self._get_eval_data(self._validation_dataset, batch_size)

    def test_data(self, batch_size):
        return self._get_eval_data(self._test_dataset, batch_size)

    def _get_eval_data(self, dataset, batch_size):
        return (
            dataset.repeat()
            .map(self._map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
