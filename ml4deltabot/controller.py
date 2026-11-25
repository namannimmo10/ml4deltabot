import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

from . import config as cfg
from .data import load_all_controller_files, split_files
from .model import build_controller_mlp, make_group_mae_metrics
from .evaluate import evaluate_predictions
from .plotting import plot_loss_curves


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden-units", type=str, default=",".join(map(str, cfg.CONTROLLER_HIDDEN_UNITS)),
                   help="comma-separated hidden layer sizes, e.g. 32,16")
    p.add_argument("--tag", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    hidden_units = [int(u) for u in args.hidden_units.split(",")]
    tag = args.tag or f"h{'-'.join(map(str, hidden_units))}"

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.random.set_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    print(f"Run tag: {tag}")

    train_paths, test_paths = split_files(cfg.DATA_DIR, cfg.TRAIN_RATIO, cfg.SEED)

    print("\nLoading train files ...")
    X_train, Y_train, feature_names = load_all_controller_files(train_paths, "train")
    print("\nLoading test files ...")
    X_test, Y_test, _ = load_all_controller_files(test_paths, "test")

    for name, arr in [("X_train", X_train), ("Y_train", Y_train), ("X_test", X_test), ("Y_test", Y_test)]:
        if not np.isfinite(arr).all():
            bad = np.argwhere(~np.isfinite(arr))[0]
            raise ValueError(f"{name} has non-finite values, e.g. at index {bad} "
                              f"(likely a corrupted row in a source file — check parsing)")

    x_scaler = StandardScaler()
    X_train_s = x_scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = x_scaler.transform(X_test).astype(np.float32)

    y_mean = Y_train.mean(axis=0).astype(np.float32)
    y_std = Y_train.std(axis=0).astype(np.float32)
    y_std = np.where(y_std < 1e-8, 1.0, y_std)

    Y_train_n = ((Y_train - y_mean) / y_std).astype(np.float32)
    Y_test_n = ((Y_test - y_mean) / y_std).astype(np.float32)

    norm_stats_out = f"normalisation_stats_controller_{tag}.npz"
    np.savez(norm_stats_out, x_mean=x_scaler.mean_, x_std=x_scaler.scale_,
              y_mean=y_mean, y_std=y_std)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train_s, Y_train_n))
        .shuffle(buffer_size=100_000, seed=cfg.SEED)
        .batch(cfg.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test_s, Y_test_n))
        .batch(cfg.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = build_controller_mlp(X_train.shape[1], hidden_units, n_outputs=Y_train.shape[1])
    model.summary()

    volt_names = ["x_in", "y_in", "z_in"]
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.CONTROLLER_LR, clipnorm=cfg.CLIPNORM),
        loss="mae",
        metrics=make_group_mae_metrics(y_std, volt_names),
    )

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=cfg.LR_PATIENCE,
            min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.ES_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            f"best_controller_{tag}.keras", monitor="val_loss", save_best_only=True, verbose=0,
        ),
    ]

    print("\nTraining ...")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=cfg.EPOCHS,
        callbacks=callbacks,
    )

    Y_pred = model.predict(X_test_s, batch_size=cfg.BATCH_SIZE, verbose=0) * y_std + y_mean
    evaluate_predictions(Y_test, Y_pred, volt_names, label="test")

    plot_loss_curves(history, f"controller_loss_curve_{tag}.png")

    model_out = f"controller_{tag}.keras"
    model.save(model_out)
    print(f"Saved {model_out} and {norm_stats_out}")


if __name__ == "__main__":
    main()
