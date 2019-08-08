# Zookeeper

[![Azure DevOps builds](https://img.shields.io/azure-devops/build/plumerai/larq/15.svg?logo=azure-devops)](https://plumerai.visualstudio.com/larq/_build/latest?definitionId=15&branchName=master) [![Azure DevOps coverage](https://img.shields.io/azure-devops/coverage/plumerai/larq/15.svg?logo=azure-devops)](https://plumerai.visualstudio.com/larq/_build/latest?definitionId=15&branchName=master) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zookeeper.svg)](https://pypi.org/project/zookeeper/) [![PyPI](https://img.shields.io/pypi/v/zookeeper.svg)](https://pypi.org/project/zookeeper/) [![PyPI - License](https://img.shields.io/pypi/l/zookeeper.svg)](https://github.com/plumerai/zookeeper/blob/master/LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black) [![Join the community on Spectrum](https://withspectrum.github.io/badge/badge.svg)](https://spectrum.chat/larq)

A small library for managing deep learning models, hyper parameters and datasets designed to make training deep learning models easy and reproducible.

## Getting Started

Zookeeper allows you to build command line interfaces for training deep learning models with very little boiler plate using [click](https://click.palletsprojects.com/) and [TensorFlow Datasets](https://www.tensorflow.org/datasets/). It helps you structure your machine learning projects in a framework agnostic and effective way.
Zookeeper is heavily inspired by [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) and [Fairseq](https://github.com/pytorch/fairseq/) but is designed to be used as a library making it lightweight and very flexible.

### Installation

```console
pip install zookeeper
pip install colorama  # optional for colored console output
```

### Registry

Zookeeper keeps track of data preprocessing, models and hyperparameters to allow you to reference them by name from the commandline.

#### Datasets and Preprocessing

TensorFlow Datasets provides [many popular datasets](https://www.tensorflow.org/datasets/datasets) that can be downloaded automatically.
In the following we will use [MNIST](http://yann.lecun.com/exdb/mnist) and define a `default` preprocessing for the images that scales the image to `[0, 1]` and uses one-hot encoding for the class labels:

```python
import tensorflow as tf
from zookeeper import cli, build_train, HParams, registry, Preprocessing

class ImageClassification(Preprocessing):
    @property
    def kwargs(self):
        return {
            "input_shape": self.features["image"].shape,
            "num_classes": self.features["label"].num_classes,
        }

    def inputs(self, data):
        return tf.cast(data["image"], tf.float32)

    def outputs(self, data):
        return tf.one_hot(data["label"], self.features["label"].num_classes)


@registry.register_preprocess("mnist")
class default(ImageClassification):
    def inputs(self, data):
        return super().inputs(data) / 255
```

#### Models

Next we will register a model called `cnn`. We will use the [Keras API](https://keras.io) for this:

```python
@registry.register_model
def cnn(hp, input_shape, num_classes):
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                hp.filters[0], (3, 3), activation=hp.activation, input_shape=input_shape
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(hp.filters[1], (3, 3), activation=hp.activation),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(hp.filters[2], (3, 3), activation=hp.activation),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hp.filters[3], activation=hp.activation),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
```

#### Hyperparameters

For each model we can register one or more hyperparameters sets that will be passed to the model function when called:

```python
@registry.register_hparams(cnn)
class basic(HParams):
    activation = "relu"
    batch_size = 32
    filters = [64, 64, 64, 64]
    learning_rate = 1e-3

    @property
    def optimizer(self):
        return tf.keras.optimizers.Adam(self.learning_rate)
```

### Training loop

To train the models registered above we will need to write a custom training loop. Zookeeper will then tie everything together:

```python
@cli.command()
@build_train()
def train(build_model, dataset, hparams, output_dir):
    """Start model training."""
    model = build_model(hparams, **dataset.preprocessing.kwargs)
    model.compile(
        optimizer=hparams.optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy", "top_k_categorical_accuracy"],
    )

    model.fit(
        dataset.train_data(hparams.batch_size),
        steps_per_epoch=dataset.train_examples // hparams.batch_size,
        validation_data=dataset.validation_data(hparams.batch_size),
        validation_steps=dataset.validation_examples // hparams.batch_size,
    )
```

This will register Click command called `train` which can be executed from the command line.

### Command Line Interface

To make the file we just created executable we will add the following lines at the bottom:

```python
if __name__ == "__main__":
    cli()
```

If you want to register your models in separate files, make sure to import them before calling `cli` to allow zookeeper to properly register them. To install your CLI as a executable command checkout the [`setuptools` integration](http://click.palletsprojects.com/en/7.x/setuptools/) of Click.

#### Usage

Zookeeper already ships with `prepare`, `plot`, and `tensorboard` commands, but now also includes the `train` command we created above:

```console
python examples/train.py --help
```

```console
Usage: train.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  install-completion  Install shell completion.
  plot                Plot data examples.
  prepare             Downloads and prepares datasets for reading.
  tensorboard         Start TensorBoard to monitor model training.
  train               Start model training.
```

To train the model we just registered run:

```console
python examples/train.py train cnn --dataset mnist --hparams-set basic --hparams batch_size=64
```

Multiple arguments are seperated by a comma, and strings should be passed without quotion marks:

```console
python examples/train.py train cnn --dataset mnist --hparams-set basic --hparams batch_size=32,actvation=relu
```
