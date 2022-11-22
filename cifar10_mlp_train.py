
"""Train an MLP on Cifar10 on one random seed. Serialize the model for
interpolation downstream."""
import argparse

import augmax
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

from flax import linen as nn
from flax.training.train_state import TrainState
from jax import jit, random, tree_map, value_and_grad, vmap
from tqdm import tqdm


tf.config.set_visible_devices([], "GPU")

activation = nn.relu


class MLPModel(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = jnp.reshape(x, (-1, 32 * 32, 3))
        x = nn.Dense(512)(x)
        x = activation(x)
        x = nn.Dense(512)(x)
        x = activation(x)
        x = nn.Dense(512)(x)
        x = activation(x)
        x = nn.Dense(10)(x)
        x = nn.log_softmax(x)
        return x


def make_stuff(model):
    train_transform = augmax.Chain(
        # augmax does not seem to support random crops with padding. See https://github.com/khdlr/augmax/issues/6.
        augmax.RandomSizedCrop(32, 32, zoom_range=(0.8, 1.2)),
        augmax.HorizontalFlip(),
        augmax.Rotate(),
    )
    # Applied to all input images, test and train.
    normalize_transform = augmax.Chain(
        augmax.ByteToFloat(), augmax.Normalize())

    @jit
    def batch_eval(params, images_u8, labels):
        images_f32 = vmap(normalize_transform)(None, images_u8)
        y_onehot = jax.nn.one_hot(labels, 10)
        logits = model.apply({"params": params}, images_f32)
        l = jnp.mean(optax.softmax_cross_entropy(
            logits=logits, labels=y_onehot))
        num_correct = jnp.sum(jnp.argmax(logits, axis=-1) == labels)
        return l, {"num_correct": num_correct}

    @jit
    def step(rng, train_state, images, labels):
        images_transformed = vmap(train_transform)(
            random.split(rng, images.shape[0]), images)
        (l, info), g = value_and_grad(batch_eval, has_aux=True)(train_state.params, images_transformed,
                                                                labels)
        return train_state.apply_gradients(grads=g), {"batch_loss": l, **info}

    def dataset_loss_and_accuracy(params, dataset, batch_size: int):
        num_examples = dataset["images_u8"].shape[0]
        assert num_examples % batch_size == 0
        num_batches = num_examples // batch_size
        batch_ix = jnp.arange(num_examples).reshape((num_batches, batch_size))
        # Can't use vmap or run in a single batch since that overloads GPU memory.
        losses, infos = zip(*[
            batch_eval(
                params,
                dataset["images_u8"][batch_ix[i, :], :, :, :],
                dataset["labels"][batch_ix[i, :]],
            ) for i in range(num_batches)
        ])
        return (
            jnp.sum(batch_size * jnp.array(losses)) / num_examples,
            sum(x["num_correct"] for x in infos) / num_examples,
        )

    return {
        "train_transform": train_transform,
        "normalize_transform": normalize_transform,
        "batch_eval": batch_eval,
        "step": step,
        "dataset_loss_and_accuracy": dataset_loss_and_accuracy,
    }


def load_cifar10():
    """Return the training and test datasets, as jnp.array's."""
    train_ds_images_u8, train_ds_labels = tfds.as_numpy(
        tfds.load("cifar10", split="train", batch_size=-1, as_supervised=True))
    test_ds_images_u8, test_ds_labels = tfds.as_numpy(
        tfds.load("cifar10", split="test", batch_size=-1, as_supervised=True))
    train_ds = {"images_u8": train_ds_images_u8, "labels": train_ds_labels}
    test_ds = {"images_u8": test_ds_images_u8, "labels": test_ds_labels}
    return train_ds, test_ds


if __name__ == "__main__":

    num_epochs = 100
    batch_size = 100
    seed = 12421
    optimizer = "adam"  # "sgd"
    learning_rate = 1e-3

    runs_to_collect = 2  # Stan's new stuff
    flattened_models_list = []

    for run_i in range(runs_to_collect):
        rng = random.PRNGKey(seed+run_i)
        def rngmix(rng, x): return random.fold_in(rng, hash(x))

        model = MLPModel()
        stuff = make_stuff(model)

        print("------------------------------------")
        print(f"STARTING RUN {run_i}")

        train_ds, test_ds = load_cifar10()
        print("train_ds labels hash", hash(
            np.array(train_ds["labels"]).tobytes()))
        print("test_ds labels hash", hash(
            np.array(test_ds["labels"]).tobytes()))

        num_train_examples = train_ds["images_u8"].shape[0]
        num_test_examples = test_ds["images_u8"].shape[0]
        assert num_train_examples % batch_size == 0
        print("num_train_examples", num_train_examples)
        print("num_test_examples", num_test_examples)

        if optimizer == "sgd":
            lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=1e-6,
                peak_value=learning_rate,
                warmup_steps=num_train_examples // batch_size,
                # Confusingly, `decay_steps` is actually the total number of steps,
                # including the warmup.
                decay_steps=num_epochs * (num_train_examples // batch_size),
            )
            # tx = optax.sgd(lr_schedule, momentum=0.9)
            tx = optax.chain(optax.add_decayed_weights(
                5e-4), optax.sgd(lr_schedule, momentum=0.9))

        elif optimizer == "adam":
            tx = optax.adam(learning_rate)

        train_state = TrainState.create(
            apply_fn=model.apply,
            params=model.init(rngmix(rng, "init"),
                              jnp.zeros((1, 32, 32, 3)))["params"],
            tx=tx,
        )
# INTE FÄRDIGT
