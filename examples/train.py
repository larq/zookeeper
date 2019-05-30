from zookeeper import cli, build_train, registry, HParams
import tensorflow as tf


@registry.register_preprocess("mnist")
def default(image):
    return tf.cast(image, dtype=tf.float32) / 255


@registry.register_model
def cnn(hp, dataset):
    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                hp.filters[0],
                (3, 3),
                activation=hp.activation,
                input_shape=dataset.input_shape,
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(hp.filters[1], (3, 3), activation=hp.activation),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(hp.filters[2], (3, 3), activation=hp.activation),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hp.filters[3], activation=hp.activation),
            tf.keras.layers.Dense(dataset.num_classes, activation="softmax"),
        ]
    )


@registry.register_hparams(cnn)
class basic(HParams):
    activation = "relu"
    batch_size = 32
    filters = [64, 64, 64, 64]
    learning_rate = 1e-3

    @property
    def optimizer(self):
        return tf.keras.optimizers.Adam(self.learning_rate)


@registry.register_hparams(cnn)
class small(basic):
    filters = [16, 32, 32, 32]


@cli.command()
@build_train
def train(build_model, dataset, hparams, output_dir, epochs):
    model = build_model(hparams, dataset)
    model.compile(
        optimizer=hparams.optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy", "top_k_categorical_accuracy"],
    )

    model.fit(
        dataset.train_data(hparams.batch_size),
        epochs=epochs,
        steps_per_epoch=dataset.train_examples // hparams.batch_size,
        validation_data=dataset.validation_data(hparams.batch_size),
        validation_steps=dataset.validation_examples // hparams.batch_size,
    )


if __name__ == "__main__":
    cli()
