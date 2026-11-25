import argparse
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import config as cfg
from .data import build_sequences, load_file, split_files
from .model import build_controller_mlp

# Row layout everywhere: [x_in, y_in, z_in, x_out, y_out, z_out] = [voltage(3), position(3)].
# The plant takes W rows ending at row r and predicts position at row r+1.


def build_rollout_arrays(data, window_size, horizon, stride):
    """
    For each start i (row r = i + W):
      H     rows r-W .. r-1 of real history                      (W, 6)
      p_now real position at row r                               (3,)
      p_ref real positions at rows r+1 .. r+K  (tracking target) (K, 3)
      v_rec recorded voltages at rows r .. r+K-1 (replay baseline only) (K, 3)
    """
    W, K = window_size, horizon
    starts = np.arange(0, len(data) - W - K, stride)
    H = data[starts[:, None] + np.arange(W)[None, :]]
    p_now = data[starts + W, 3:6]
    p_ref = data[starts[:, None] + W + 1 + np.arange(K)[None, :]][:, :, 3:6]
    v_rec = data[starts[:, None] + W + np.arange(K)[None, :]][:, :, 0:3]
    return tuple(a.astype(np.float32) for a in (H, p_now, p_ref, v_rec))


def load_rollout_arrays(paths, window_size, horizon, stride, offset):
    parts = [build_rollout_arrays(load_file(p, offset=offset), window_size, horizon, stride) for p in paths]
    return tuple(np.concatenate(group, axis=0) for group in zip(*parts))


def rollout(H, p_now, p_ref, v_rec, mode, ctx, training=False):
    """
    Run controller (or a baseline) and the frozen plant together for K steps.
    mode: "controller" | "replay" (recorded voltages) | "hold" (keep v(r-1)).
    The controller never sees the voltage it is choosing: its inputs are the
    required displacement, the current velocity and the previous voltage.
    """
    window = H
    v_hold = H[:, -1, 0:3]
    v_prev = v_hold
    p_prev = H[:, -1, 3:6]
    p_cur = p_now
    preds, volts = [], []
    slew, bounds = 0.0, 0.0

    for k in range(ctx.horizon):
        if mode == "controller":
            feats = tf.concat([p_ref[:, k, :] - p_cur, (p_cur - p_prev) / ctx.dt, v_prev], axis=1)
            out = ctx.controller((feats - ctx.feat_mean) / ctx.feat_std, training=training)
            v_raw = v_prev + out * ctx.dv_scale
            over = tf.nn.relu(v_raw - ctx.v_max) + tf.nn.relu(ctx.v_min - v_raw)
            bounds += tf.reduce_mean(tf.square(over / ctx.v_range))
            slew += tf.reduce_mean(tf.square(out))
            v = tf.clip_by_value(v_raw, ctx.v_min, ctx.v_max)
        elif mode == "replay":
            v = v_rec[:, k, :]
        elif mode == "hold":
            v = v_hold
        else:
            raise ValueError(mode)

        row = tf.concat([v, p_cur], axis=1)[:, tf.newaxis, :]
        window = tf.concat([window[:, 1:, :], row], axis=1)
        plant_out = ctx.plant((window - ctx.px_mean) / ctx.px_std, training=False)

        if ctx.position_mode == "delta":
            vel = plant_out[:, ctx.vel_start:ctx.vel_start + 3] * ctx.pv_std + ctx.pv_mean
            p_next = p_cur + vel * ctx.dt
        else:
            p_next = plant_out[:, ctx.pos_start:ctx.pos_start + 3] * ctx.pp_std + ctx.pp_mean

        preds.append(p_next)
        volts.append(v)
        p_prev, p_cur, v_prev = p_cur, p_next, v

    return tf.stack(preds, axis=1), tf.stack(volts, axis=1), slew / ctx.horizon, bounds / ctx.horizon


def report(mode, pos_pred, p_ref, tolerances, steps):
    err = np.abs(pos_pred - p_ref)
    print(f"\n[{mode}]  pooled MAE over all steps = {err.mean():.2f} nm")
    print(f"  {'step':>4s} {'axis':4s} {'p50(nm)':>9s} {'p95(nm)':>9s}  " +
          "  ".join(f"<={t:g}nm" for t in tolerances))
    for k in steps:
        for i, axis in enumerate("xyz"):
            e = err[:, k - 1, i]
            p50, p95 = np.percentile(e, [50, 95])
            rates = "  ".join(f"{(e <= t).mean() * 100:6.1f}%" for t in tolerances)
            print(f"  {k:4d} {axis:4s} {p50:9.2f} {p95:9.2f}  {rates}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plant-model", required=True)
    p.add_argument("--plant-norm-stats", required=True)
    p.add_argument("--window-size", type=int, default=cfg.WINDOW_SIZE)
    p.add_argument("--horizon", type=int, default=10, help="rollout steps K")
    p.add_argument("--train-stride", type=int, default=5)
    p.add_argument("--test-stride", type=int, default=5)
    p.add_argument("--hidden-units", default=",".join(map(str, cfg.CONTROLLER_HIDDEN_UNITS)))
    p.add_argument("--activation", default="relu", choices=["relu", "tanh"])
    p.add_argument("--position-mode", default="delta", choices=["delta", "direct"],
                   help="delta: p(r+1) = p(r) + plant velocity * DT.  direct: plant position head")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--target-offset", type=float, default=30.0,
                   help="nm. Shift each target by a random amount in +-this (same shift for all K steps). "
                        "Without it the recorded voltage is already the right answer and nothing can be learned.")
    p.add_argument("--dv-scale-frac", type=float, default=0.01,
                   help="one unit of controller output = this fraction of the voltage range per step")
    p.add_argument("--pos-ref", type=float, default=10.0,
                   help="nm; the position loss is expressed in units of this (10nm spec)")
    p.add_argument("--w-slew", type=float, default=0.05, help="penalty on voltage increments")
    p.add_argument("--w-bounds", type=float, default=1.0, help="penalty on leaving the voltage range")
    p.add_argument("--tolerances", default="10,20,50")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    hidden_units = [int(u) for u in args.hidden_units.split(",")]
    tolerances = [float(t) for t in args.tolerances.split(",")]
    K = args.horizon
    tag = args.tag or f"cl_h{'-'.join(map(str, hidden_units))}_{args.activation}_K{K}_{args.position_mode}"

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.random.set_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    # ---- plant (frozen) and its scalers
    plant = keras.models.load_model(args.plant_model, compile=False)
    plant.trainable = False
    stats = np.load(args.plant_norm_stats)

    _, _, names = build_sequences(np.zeros((args.window_size + 2, 6), np.float32), args.window_size)
    pos_start = names.index("x_out")
    vel_start = names.index("vx") if "vx" in names else None
    if args.position_mode == "delta" and vel_start is None:
        raise SystemExit("--position-mode delta needs a plant with a velocity head (cfg.TARGETS)")

    y_mean, y_std = stats["y_mean"], stats["y_std"]
    vs = vel_start if vel_start is not None else 0

    # All positions are held relative to the plant's own position mean, removed in
    # float64 before the float32 cast (absolute z only resolves 4 nm in float32).
    # The plant subtracts that same mean internally, so its inputs are unchanged.
    offset = np.concatenate([np.zeros(3), np.asarray(stats["x_mean"], dtype=np.float64)[3:6]])
    px_mean = np.asarray(stats["x_mean"], dtype=np.float64).copy()
    px_mean[3:6] = 0.0
    pp_mean = np.asarray(y_mean, dtype=np.float64)[pos_start:pos_start + 3] - offset[3:6]

    # ---- data
    train_paths, test_paths = split_files(cfg.DATA_DIR, cfg.TRAIN_RATIO, cfg.SEED)
    print("Loading train files ...")
    train = load_rollout_arrays(train_paths, args.window_size, K, args.train_stride, offset)
    print("Loading test files ...")
    test = load_rollout_arrays(test_paths, args.window_size, K, args.test_stride, offset)
    H_tr, pn_tr, pr_tr, _ = train
    A = args.target_offset
    rng = np.random.default_rng(cfg.SEED)
    if A > 0:
        stat_shift = rng.uniform(-A, A, size=(len(pn_tr), 3)).astype(np.float32)
        test_shift = rng.uniform(-A, A, size=(len(test[0]), 1, 3)).astype(np.float32)
        test = (test[0], test[1], test[2] + test_shift, test[3])      # fixed targets for a fair comparison
    else:
        stat_shift = 0.0
    print(f"train: {len(H_tr):,} rollouts   test: {len(test[0]):,} rollouts   (K={K}, target offset +-{A:g} nm)")

    # ---- controller input/output scaling, voltage limits (all from training data)
    dt = cfg.DT
    feats0 = np.concatenate([pr_tr[:, 0, :] + stat_shift - pn_tr, (pn_tr - H_tr[:, -1, 3:6]) / dt, H_tr[:, -1, 0:3]], axis=1)
    feat_mean = feats0.mean(axis=0)
    feat_std = np.where(feats0.std(axis=0) < 1e-8, 1.0, feats0.std(axis=0))
    dv = H_tr[:, -1, 0:3] - H_tr[:, -2, 0:3]
    dv_std = np.where(dv.std(axis=0) < 1e-8, 1.0, dv.std(axis=0))
    volts = H_tr[:, :, 0:3].reshape(-1, 3)
    v_min, v_max = volts.min(axis=0), volts.max(axis=0)
    v_range = np.maximum(v_max - v_min, 1e-8)
    dv_scale = np.maximum(dv_std, args.dv_scale_frac * v_range)

    if args.position_mode == "delta":
        implied = y_std[vs:vs + 3] * dt
        measured = np.std(H_tr[:, -1, 3:6] - H_tr[:, -2, 3:6], axis=0)
        if np.any(implied / measured > 2) or np.any(implied / measured < 0.5):
            print(f"WARNING: plant velocity std * cfg.DT = {implied} nm/step but measured displacement "
                  f"std = {measured} nm/step. cfg.DT probably differs from the plant's training value; "
                  f"delta mode will be wrong. Run diagnose_plant first.")

    c = lambda a: tf.constant(np.asarray(a, dtype=np.float32))
    controller = build_controller_mlp(feats0.shape[1], hidden_units, n_outputs=3,
                                      activation=args.activation, zero_init_output=True)
    controller.summary()

    ctx = SimpleNamespace(
        controller=controller, plant=plant, horizon=K, dt=dt, position_mode=args.position_mode,
        feat_mean=c(feat_mean), feat_std=c(feat_std), dv_scale=c(dv_scale),
        v_min=c(v_min), v_max=c(v_max), v_range=c(v_range),
        px_mean=c(px_mean), px_std=c(stats["x_std"]),
        pos_start=pos_start, vel_start=vs,
        pp_mean=c(pp_mean), pp_std=c(y_std[pos_start:pos_start + 3]),
        pv_mean=c(y_mean[vs:vs + 3]), pv_std=c(y_std[vs:vs + 3]),
    )

    steps_per_epoch = len(H_tr) // args.batch_size
    schedule = keras.optimizers.schedules.CosineDecay(args.lr, steps_per_epoch * args.epochs, alpha=0.02)
    optimizer = keras.optimizers.Adam(schedule, clipnorm=1.0)
    optimizer.build(controller.trainable_variables)

    @tf.function
    def train_step(H, pn, pr, vr):
        if A > 0:
            pr = pr + tf.random.uniform(tf.shape(pn), minval=-A, maxval=A)[:, tf.newaxis, :]
        with tf.GradientTape() as tape:
            pred, _, slew, bounds = rollout(H, pn, pr, vr, "controller", ctx, training=True)
            pos_err = tf.reduce_mean(tf.abs(pred - pr))
            loss = pos_err / args.pos_ref + args.w_slew * slew + args.w_bounds * bounds
        grads = tape.gradient(loss, controller.trainable_variables)
        optimizer.apply_gradients(zip(grads, controller.trainable_variables))
        return pos_err, slew, bounds

    @tf.function
    def simulate(H, pn, pr, vr, mode):
        pred, _, _, _ = rollout(H, pn, pr, vr, mode, ctx, training=False)
        return pred

    def dataset(arrays, shuffle, drop):
        ds = tf.data.Dataset.from_tensor_slices(arrays)
        if shuffle:
            ds = ds.shuffle(100_000, seed=cfg.SEED)
        return ds.batch(args.batch_size, drop_remainder=drop).prefetch(tf.data.AUTOTUNE)

    train_ds, test_ds = dataset(train, True, True), dataset(test, False, False)

    def collect(mode):
        return np.concatenate([simulate(*b, mode).numpy() for b in test_ds], axis=0)

    print(f"\nTraining: {K}-step closed loop through the frozen plant, position mode = {args.position_mode}")
    best, bad = float("inf"), 0
    for epoch in range(args.epochs):
        tr = []
        for batch in train_ds:
            pos_err, slew, bounds = train_step(*batch)
            if not np.isfinite(float(pos_err)):
                raise SystemExit("non-finite loss: lower --lr, or check the plant/DT with diagnose_plant")
            tr.append((float(pos_err), float(slew), float(bounds)))
        tr = np.mean(tr, axis=0)
        val = np.abs(collect("controller") - test[2]).mean()
        print(f"Epoch {epoch + 1:3d}/{args.epochs}  train_pos_mae={tr[0]:9.2f}nm  val_pos_mae={val:9.2f}nm  "
              f"slew={tr[1]:.4f}  bounds={tr[2]:.5f}")

        if val < best:
            best, bad = val, 0
            controller.save(f"controller_{tag}.keras")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    ctx.controller = keras.models.load_model(f"controller_{tag}.keras", compile=False)
    steps = sorted({1, K})
    results = {}
    for mode in ("hold", "replay", "controller"):
        results[mode] = collect(mode)
        report(mode, results[mode], test[2], tolerances, steps)
    mae = {m: np.abs(r - test[2]).mean() for m, r in results.items()}
    print(f"\nController vs hold-the-voltage: {mae['controller']:.2f} nm vs {mae['hold']:.2f} nm "
          f"({100 * (mae['hold'] - mae['controller']) / mae['hold']:.1f}% lower).  "
          f"If this is near 0%, the controller has learned nothing beyond the plant's own error.")

    np.savez(f"controller_{tag}_stats.npz", feat_mean=feat_mean, feat_std=feat_std, dv_scale=dv_scale,
             v_min=v_min, v_max=v_max, dt=dt, position_offset=offset[3:6])
    print(f"\nSaved controller_{tag}.keras and controller_{tag}_stats.npz (best val_pos_mae={best:.2f}nm)")
    print("Baselines: hold = keep the last voltage; replay = feed the plant the recorded voltages.")


if __name__ == "__main__":
    main()
