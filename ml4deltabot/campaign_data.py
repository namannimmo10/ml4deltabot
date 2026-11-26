"""
Loader for the October 2026 campaign HDF5 captures.

Produces exactly the same (N, 6) float32 contract as data.load_file:
    [x_in, y_in, z_in, x_out, y_out, z_out]
     ^ command counts      ^ position nm
so build_sequences / build_rollout_arrays / build_controller_features all
work unchanged.

Facts this encodes (verified against the captures, see deltabot_characterisation.md):

  * Position lives in INENC1/2/3.VAL.Value.  Campaign/band/profile captures
    store RAW ENCODER COUNTS (x 0.256 -> nm).  Noise/drive captures are
    already in nm and say so via the `janus_capture_scales` root attribute.
    We detect that attribute rather than hardcoding per directory.

  * Command lives in PGEN*.OUT.Value.  Single-axis captures have only
    PGEN1, which drives whichever axis the FILENAME names; the other two
    axes are commanded to zero.  MIMO captures have PGEN1/2/3 -> x/y/z.

  * PGEN1 includes the DC offset for the `positions` segment, so the
    command column is complete as-is.

  * Captures with a non-zero PID.REAL_OUTPUT were taken with the loop
    CLOSED: there PGEN is a setpoint, not the amplifier command, so the
    (command, position) pair means something different.  Excluded by
    default from open-loop plant training.

  * Capture rate is 40 kHz; the command rate is 20 kHz.  Decimation is
    anti-alias filtered, never plain striding -- the 850-1050 Hz
    structural modes would otherwise fold into the scan band.
"""
import os
import re
import glob

import numpy as np
import h5py
from scipy import signal

from . import config as cfg

CAPTURE_RATE_HZ = 40000.0
ENCODER_NM_PER_COUNT = 0.256
POS_KEYS = ("INENC1.VAL.Value", "INENC2.VAL.Value", "INENC3.VAL.Value")
CMD_KEYS = ("PGEN1.OUT.Value", "PGEN2.OUT.Value", "PGEN3.OUT.Value")

# segments whose (command, position) pair is not an open-loop plant observation
EXCLUDE_SEGMENTS = ("quiet_start", "quiet_end", "closed_loop")

_AXIS_RE = re.compile(r"^([xyz]+)_")


def axes_in_filename(path):
    """'x_l1_r2.h5' -> ('x',);  'xyz_l1_r1.h5' -> ('x','y','z')."""
    m = _AXIS_RE.match(os.path.basename(path))
    if not m:
        return ()
    return tuple(m.group(1))


def _is_closed_loop(f):
    if "PID.REAL_OUTPUT.Value" not in f:
        return False
    d = f["PID.REAL_OUTPUT.Value"]
    probe = np.asarray(d[::max(1, len(d) // 20000)], dtype=np.float64)
    return bool(np.any(probe != 0.0))


def load_campaign_file(path, offset=None, target_rate_hz=None, include_closed_loop=False):
    """
    One capture -> (N, 6) float32 [x_in,y_in,z_in,x_out,y_out,z_out].

    offset: length-6 array subtracted in float64 BEFORE the float32 cast
            (same role as in data.load_file; z sits near -4.8e7 nm where
            float32 only resolves 4 nm).
    target_rate_hz: anti-alias filter + decimate to this rate. Defaults to
            1/cfg.DT. Must divide 40 kHz exactly.
    Returns None if the capture should be skipped.
    """
    target_rate_hz = target_rate_hz or (1.0 / cfg.DT)
    q = CAPTURE_RATE_HZ / target_rate_hz
    if abs(q - round(q)) > 1e-9:
        raise ValueError(f"{CAPTURE_RATE_HZ} Hz does not divide by {target_rate_hz} Hz")
    q = int(round(q))

    with h5py.File(path, "r") as f:
        if not include_closed_loop and _is_closed_loop(f):
            return None
        if not all(k in f for k in POS_KEYS):
            return None

        # already-nm captures advertise the scale they were written with
        already_nm = "janus_capture_scales" in f.attrs
        pos_scale = 1.0 if already_nm else ENCODER_NM_PER_COUNT

        pos = np.stack([np.asarray(f[k][:], dtype=np.float64) * pos_scale
                        for k in POS_KEYS], axis=1)

        cmd = np.zeros_like(pos)
        present = [k for k in CMD_KEYS if k in f]
        driven = axes_in_filename(path)
        if len(present) >= 3:
            for i, k in enumerate(CMD_KEYS):
                cmd[:, i] = np.asarray(f[k][:], dtype=np.float64)
        elif len(present) == 1 and len(driven) == 1:
            cmd[:, "xyz".index(driven[0])] = np.asarray(f[present[0]][:], dtype=np.float64)
        elif len(present) == 1 and len(driven) != 1:
            raise ValueError(
                f"{path}: one PGEN channel but filename names {driven or 'no'} axes; "
                "cannot tell which axis the command drives")
        # no PGEN at all (pure noise-floor capture): command stays zero

    data = np.concatenate([cmd, pos], axis=1)

    if q > 1:
        # FIR, zero-phase: no group delay, so command and position stay aligned.
        n = (len(data) // q) * q
        data = signal.decimate(data[:n], q, ftype="fir", zero_phase=True, axis=0)

    if offset is not None:
        data = data - np.asarray(offset, dtype=np.float64)
    return data.astype(np.float32)


def find_campaign_files(root, exclude_segments=EXCLUDE_SEGMENTS):
    """All usable .h5 captures under a campaign run directory."""
    paths = sorted(glob.glob(os.path.join(root, "**", "*.h5"), recursive=True))
    keep = []
    for p in paths:
        rel = os.path.relpath(p, root)
        seg = rel.split(os.sep)[0]
        if seg in exclude_segments:
            continue
        if not axes_in_filename(p) and "multisine" not in os.path.basename(p):
            continue
        keep.append(p)
    if not keep:
        raise FileNotFoundError(f"no usable .h5 captures under {root!r}")
    return keep


def split_campaign_runs(campaign_root, train_runs=None, val_runs=None):
    """
    Split by CAMPAIGN RUN, not by random file.

    The two October runs are independent repeats an hour apart that agree to
    1.000-1.006 in gain, so holding one out measures session generalisation
    honestly. A random file shuffle can put two captures of the same
    excitation on opposite sides of the split and flatter the model.
    """
    # islink check matters: the rig writes a `latest` symlink pointing at one
    # of the real runs. Including it would put the same captures in both
    # halves of the split.
    runs = sorted(d for d in glob.glob(os.path.join(campaign_root, "*"))
                  if os.path.isdir(d) and not os.path.islink(d))
    if len(runs) < 2:
        raise ValueError(f"need >=2 campaign runs under {campaign_root!r}, found {len(runs)}")
    train_runs = train_runs or runs[:-1]
    val_runs = val_runs or runs[-1:]
    train = [p for r in train_runs for p in find_campaign_files(r)]
    val = [p for r in val_runs for p in find_campaign_files(r)]
    return train, val


def describe(paths, target_rate_hz=None):
    """Print what you are about to train on. Cheap sanity check."""
    target_rate_hz = target_rate_hz or (1.0 / cfg.DT)
    print(f"{'capture':<44s} {'rows':>10s} {'cmd rms':>9s} {'pos rms um':>11s} {'pos mean um':>12s}")
    tot = 0
    for p in paths:
        d = load_campaign_file(p, target_rate_hz=target_rate_hz)
        if d is None:
            print(f"{os.path.relpath(p):<44s}  skipped (closed loop)")
            continue
        tot += len(d)
        print(f"{os.path.relpath(p):<44s} {len(d):>10,d} "
              f"{np.abs(d[:, 0:3]).max():>9.0f} "
              f"{d[:, 3:6].std(axis=0).max()/1000:>11.1f} "
              f"{d[:, 3:6].mean(axis=0).max()/1000:>12.1f}")
    print(f"{'TOTAL':<44s} {tot:>10,d} rows at {target_rate_hz:.0f} Hz "
          f"({tot/target_rate_hz/60:.1f} min)")
