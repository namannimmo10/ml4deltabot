import os

DATA_DIR = "/dls/science/users/qfc83269/deltabot_data"

WINDOW_SIZE = 10
GRU_UNITS = 64
DENSE_UNITS = 64
BATCH_SIZE = 1024 * 4
EPOCHS = 500
LR = 1e-3
CLIPNORM = 1.0
TRAIN_RATIO = 0.8
SEED = 42
ES_PATIENCE = 40
LR_PATIENCE = 15
STEPS_PER_EXECUTION = 100

# Column indices in the raw txt file (10 columns total)
COL = dict(x_in=1, y_in=3, z_in=5, x_out=7, y_out=8, z_out=9)

# Which quantities the plant model predicts at the next timestep.
# Voltage/demand is supplied by the controller model at each step, not
# predicted by the plant — the plant only needs to forecast the physical
# response (position, velocity) to a given voltage.
TARGETS = ("position", "velocity")

# Sample interval between rows, used to convert position deltas to velocity.
# Confirmed sample rate: 0.0001s per row.
DT = 0.0001

# Acceptance thresholds for the "% within tolerance" metric, in physical
# units per target group. Position: the 10nm spec directly. Velocity: no
# independent spec was given, so this is derived from the position spec —
# a velocity error this size corresponds to a 10nm position error over one
# sample (10nm / DT). Treat as a starting point, not a hard requirement.
TOLERANCES = dict(position=10.0, velocity=10.0 / DT)

# Display-only unit conversion for metric printouts — native units (nm,
# nm/s) stay unchanged everywhere else (training, tolerance comparison).
# (label, multiply-by-this-to-convert-native-to-display-unit)
DISPLAY_UNITS = dict(position=("nm", 1.0), velocity=("um/s", 1e-3))

MODEL_OUT = "gru_64.keras"
CHECKPOINT_OUT = "best_gru.keras"
NORM_STATS_OUT = "normalisation_stats_gru.npz"
LOSS_CURVE_OUT = "gru_loss_curve.png"

# Controller (inverse model) — small feedforward net, no recurrence.
CONTROLLER_HIDDEN_UNITS = [32, 16]
CONTROLLER_LR = 1e-4
CONTROLLER_MODEL_OUT = "controller_mlp.keras"
CONTROLLER_CHECKPOINT_OUT = "best_controller.keras"
CONTROLLER_NORM_STATS_OUT = "normalisation_stats_controller.npz"
CONTROLLER_LOSS_CURVE_OUT = "controller_loss_curve.png"
