import numpy as np
import tensorflow_datasets as tfds
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_examples(dataset):
    fig, axes = plt.subplots(2, 3)
    for ax, image in zip(axes.reshape(-1), tfds.as_numpy(dataset)):
        if np.issubdtype(image.dtype, np.floating) and image.ndim == 3:
            im = ax.imshow(image.mean(axis=-1), aspect="equal", cmap="gray")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        else:
            if image.shape[-1] == 3:
                ax.imshow(image, aspect="equal")
            else:
                ax.imshow(image.squeeze(), aspect="equal", cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    return fig


def plot_all_examples(set):
    dataset = set.load_split(set.train_split, shuffle=False)
    decoder = set.info.features["image"].decode_example

    raw = plot_examples(dataset.map(lambda feat: decoder(feat["image"])))
    train = plot_examples(dataset.map(lambda feat: set.map_fn(feat, training=True)[0]))
    eval = plot_examples(dataset.map(lambda feat: set.map_fn(feat)[0]))
    return raw, train, eval
