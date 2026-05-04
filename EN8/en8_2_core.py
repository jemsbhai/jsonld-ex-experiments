"""EN8.2 — IoT Sensor Pipeline: Core Module.

Provides data loading for the Intel Berkeley Research Lab dataset,
time alignment, and six sensor fusion methods for comparison:
  1. Scalar average (equal weight)
  2. Inverse-variance weighted average
  3. 1D Kalman filter
  4. Gaussian Process regression
  5. Subjective Logic fusion
  6. Dempster-Shafer fusion
  7. SL fusion with temporal decay

References:
    Bodik et al. (2004). Intel Lab sensor data.
    Jøsang (2016). Subjective Logic. Springer.
    Rasmussen & Williams (2006). Gaussian Processes for ML. MIT Press.
"""
from __future__ import annotations

import gzip
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
)
from jsonld_ex.confidence_decay import decay_opinion

# DS comparison operators (from EN4.2)
import sys
_EXPERIMENTS_ROOT = Path(__file__).resolve().parent.parent
if str(_EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_ROOT))
from EN4.en4_2_ds_comparison import (
    dempster_combine,
    compute_conflict_K,
)

# ── Data paths ──────────────────────────────────────────────────────
_EN8_DIR = Path(__file__).resolve().parent
_DATA_DIR = _EN8_DIR / "data"
_INTEL_DATA = _DATA_DIR / "intel_lab_data.txt.gz"
_MOTE_LOCS = _DATA_DIR / "mote_locs.txt"


# =====================================================================
# 1. Data Loading
# =====================================================================

def load_intel_lab_data(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load Intel Berkeley Research Lab sensor data.

    Schema: date, time, epoch, moteid, temperature, humidity, light, voltage

    Returns:
        List of dicts with keys: timestamp, epoch, moteid,
        temperature, humidity, light, voltage.
    """
    path = path or _INTEL_DATA
    records = []

    open_fn = gzip.open if str(path).endswith(".gz") else open

    with open_fn(path, "rt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                date_str = parts[0]
                time_str = parts[1]
                epoch = int(parts[2])
                moteid = int(parts[3])
                temperature = float(parts[4])
                humidity = float(parts[5])
                light = float(parts[6])
                voltage = float(parts[7])

                # Parse timestamp
                ts_str = f"{date_str} {time_str}"
                # Handle variable precision in seconds
                try:
                    ts = datetime.strptime(ts_str[:23], "%Y-%m-%d %H:%M:%S.%f")
                except ValueError:
                    ts = datetime.strptime(ts_str[:19], "%Y-%m-%d %H:%M:%S")

                records.append({
                    "timestamp": ts.timestamp(),
                    "epoch": epoch,
                    "moteid": moteid,
                    "temperature": temperature,
                    "humidity": humidity,
                    "light": light,
                    "voltage": voltage,
                })
            except (ValueError, IndexError):
                continue  # skip malformed lines

    return records


def load_mote_locations(path: Optional[Path] = None) -> Dict[int, Tuple[float, float]]:
    """Load sensor (mote) physical locations.

    Returns:
        Dict mapping moteid -> (x, y) in meters.
    """
    path = path or _MOTE_LOCS
    locs = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                mid = int(parts[0])
                x, y = float(parts[1]), float(parts[2])
                locs[mid] = (x, y)
    return locs


def select_sensor_cluster(
    data: List[Dict[str, Any]],
    sensor_ids: List[int],
) -> List[Dict[str, Any]]:
    """Filter data to only include specified sensors."""
    id_set = set(sensor_ids)
    return [r for r in data if r["moteid"] in id_set]


# =====================================================================
# 2. Time Alignment
# =====================================================================

def time_align_readings(
    data: List[Dict[str, Any]],
    sensor_ids: List[int],
    interval_seconds: int = 300,
    field: str = "temperature",
) -> Dict[str, Any]:
    """Align asynchronous sensor readings to a uniform time grid.

    For each time interval, takes the reading closest to the interval
    midpoint. If no reading within the interval, assigns NaN.

    Args:
        data: Filtered sensor data (from select_sensor_cluster).
        sensor_ids: List of sensor IDs to align.
        interval_seconds: Grid spacing in seconds (default 5 min).
        field: Which measurement to align (default temperature).

    Returns:
        Dict with 'timestamps' (array of epoch seconds) and
        sensor_id -> array of values (NaN for missing).
    """
    if not data:
        return {"timestamps": np.array([])}

    # Group by sensor
    by_sensor: Dict[int, List[Tuple[float, float]]] = {s: [] for s in sensor_ids}
    for r in data:
        sid = r["moteid"]
        if sid in by_sensor and r[field] is not None and not math.isnan(r[field]):
            by_sensor[sid].append((r["timestamp"], r[field]))

    # Sort each sensor by time
    for sid in by_sensor:
        by_sensor[sid].sort()

    # Determine time range
    all_times = [t for sid in by_sensor for t, _ in by_sensor[sid]]
    if not all_times:
        return {"timestamps": np.array([])}

    t_min = min(all_times)
    t_max = max(all_times)

    # Build time grid
    n_steps = int((t_max - t_min) / interval_seconds) + 1
    timestamps = np.array([t_min + i * interval_seconds for i in range(n_steps)])

    result: Dict[str, Any] = {"timestamps": timestamps}

    for sid in sensor_ids:
        values = np.full(n_steps, np.nan)
        readings = by_sensor.get(sid, [])
        if not readings:
            result[sid] = values
            continue

        # For each grid point, find closest reading within interval
        read_times = np.array([t for t, _ in readings])
        read_vals = np.array([v for _, v in readings])

        for i, t_grid in enumerate(timestamps):
            # Find readings within this interval
            mask = np.abs(read_times - t_grid) <= interval_seconds / 2
            if np.any(mask):
                # Take the closest one
                dists = np.abs(read_times[mask] - t_grid)
                closest_idx = np.argmin(dists)
                values[i] = read_vals[mask][closest_idx]

        result[sid] = values

    return result


# =====================================================================
# 3. Fusion Methods
# =====================================================================

def scalar_average(readings: np.ndarray) -> float:
    """Simple mean of available readings (ignoring NaN)."""
    valid = readings[~np.isnan(readings)]
    if len(valid) == 0:
        return float("nan")
    return float(np.mean(valid))


def weighted_average(readings: np.ndarray, sigmas: np.ndarray) -> float:
    """Inverse-variance weighted average.

    Args:
        readings: Array of sensor values (may contain NaN).
        sigmas: Array of sensor noise std devs (same length).

    Returns:
        Weighted average, or NaN if no valid readings.
    """
    valid = ~np.isnan(readings)
    if not np.any(valid):
        return float("nan")
    r = readings[valid]
    s = sigmas[valid]
    weights = 1.0 / (s ** 2)
    return float(np.sum(weights * r) / np.sum(weights))


def kalman_fuse_1d(
    readings: np.ndarray,
    sigmas: np.ndarray,
    process_noise: float = 0.01,
) -> np.ndarray:
    """1D Kalman filter for multi-sensor fusion.

    State: scalar estimate of the true value.
    Observation: multiple sensor readings per time step.

    Args:
        readings: (n_steps, n_sensors) array. NaN = missing.
        sigmas: (n_sensors,) array of sensor noise std devs.
        process_noise: State transition noise variance (Q).

    Returns:
        (n_steps,) array of fused estimates.
    """
    n_steps, n_sensors = readings.shape
    estimates = np.zeros(n_steps)

    # Initialize from first valid reading
    for i in range(n_steps):
        valid = ~np.isnan(readings[i])
        if np.any(valid):
            x = np.mean(readings[i, valid])
            break
    else:
        return np.full(n_steps, np.nan)

    P = 1.0  # initial state variance

    for t in range(n_steps):
        # Predict
        x_pred = x
        P_pred = P + process_noise

        # Update with each available sensor
        for s in range(n_sensors):
            if np.isnan(readings[t, s]):
                continue
            R = sigmas[s] ** 2  # measurement noise variance
            K = P_pred / (P_pred + R)  # Kalman gain
            x_pred = x_pred + K * (readings[t, s] - x_pred)
            P_pred = (1 - K) * P_pred

        x = x_pred
        P = P_pred
        estimates[t] = x

    return estimates


def gp_fuse_1d(
    readings: np.ndarray,
    sigmas: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian Process regression for multi-sensor fusion.

    Treats all sensor readings as noisy observations of a latent
    function. Uses RBF kernel with noise variance per sensor.

    Args:
        readings: (n_steps, n_sensors) array. NaN = missing.
        sigmas: (n_sensors,) array of sensor noise std devs.

    Returns:
        (estimates, uncertainties) — both (n_steps,) arrays.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    n_steps, n_sensors = readings.shape

    # Build training data: (time_index, sensor_index) -> value
    X_train = []
    y_train = []
    noise_per_point = []

    for t in range(n_steps):
        for s in range(n_sensors):
            if not np.isnan(readings[t, s]):
                X_train.append([float(t)])
                y_train.append(readings[t, s])
                noise_per_point.append(sigmas[s] ** 2)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    avg_noise = np.mean(noise_per_point)

    # Fit GP
    kernel = RBF(length_scale=5.0) + WhiteKernel(noise_level=avg_noise)
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=np.array(noise_per_point),
        n_restarts_optimizer=2, normalize_y=True,
    )
    gp.fit(X_train, y_train)

    # Predict at each time step
    X_pred = np.arange(n_steps).reshape(-1, 1).astype(float)
    y_pred, y_std = gp.predict(X_pred, return_std=True)

    return y_pred, y_std


# =====================================================================
# 4. SL and DS Fusion for Continuous Values
# =====================================================================

def _sensor_to_opinion(
    value: float,
    sigma: float,
    value_range: Tuple[float, float],
    base_rate: float = 0.5,
) -> Opinion:
    """Convert a sensor reading to an SL opinion.

    Maps sensor reliability (inverse noise) to opinion uncertainty.
    Higher noise → higher uncertainty → less evidence.

    The 'confidence' represents how reliable this reading is:
        confidence = 1 - clamp(sigma / range_width, 0, 1)
        uncertainty = clamp(sigma / range_width * 4, 0.02, 0.95)
    """
    range_width = value_range[1] - value_range[0]
    if range_width <= 0:
        range_width = 1.0

    # Uncertainty proportional to noise relative to value range
    u = max(0.02, min(0.95, sigma / range_width * 4))

    # Confidence: normalized position in range
    norm_val = (value - value_range[0]) / range_width
    norm_val = max(0.001, min(0.999, norm_val))

    return Opinion.from_confidence(norm_val, uncertainty=u)


def sl_fuse_sensors(
    readings: np.ndarray,
    sigmas: np.ndarray,
    value_range: Tuple[float, float],
    base_rate: float = 0.5,
) -> np.ndarray:
    """Fuse sensor readings using SL cumulative fusion.

    For each time step:
    1. Build SL opinion per sensor (reliability from sigma)
    2. Cumulative-fuse all available opinions
    3. Estimate = opinion-weighted average of readings

    Args:
        readings: (n_steps, n_sensors) or (n_steps,) for 1 step.
        sigmas: (n_sensors,) noise std devs.
        value_range: (min, max) of plausible values.

    Returns:
        (n_steps,) array of fused estimates.
    """
    if readings.ndim == 1:
        readings = readings.reshape(1, -1)

    n_steps, n_sensors = readings.shape
    estimates = np.zeros(n_steps)

    for t in range(n_steps):
        valid_mask = ~np.isnan(readings[t])
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            estimates[t] = np.nan
            continue
        if len(valid_indices) == 1:
            estimates[t] = readings[t, valid_indices[0]]
            continue

        # Build opinions and fuse
        opinions = []
        for idx in valid_indices:
            op = _sensor_to_opinion(
                readings[t, idx], sigmas[idx], value_range, base_rate)
            opinions.append(op)

        fused = opinions[0]
        for op in opinions[1:]:
            fused = cumulative_fuse(fused, op)

        # Weight by evidence strength (1 - uncertainty)
        # NOT projected_probability, which conflates position with reliability
        weights = []
        for op in opinions:
            weights.append(1.0 - op.uncertainty)
        weights = np.array(weights)
        w_sum = np.sum(weights)
        if w_sum > 0:
            estimates[t] = np.sum(
                weights * readings[t, valid_indices]) / w_sum
        else:
            estimates[t] = np.mean(readings[t, valid_indices])

    return estimates


def ds_fuse_sensors(
    readings: np.ndarray,
    sigmas: np.ndarray,
    value_range: Tuple[float, float],
    base_rate: float = 0.5,
) -> np.ndarray:
    """Fuse sensor readings using classical Dempster's rule.

    Same approach as sl_fuse_sensors but uses DS combination
    instead of SL cumulative fusion for weight computation.
    """
    if readings.ndim == 1:
        readings = readings.reshape(1, -1)

    n_steps, n_sensors = readings.shape
    estimates = np.zeros(n_steps)

    for t in range(n_steps):
        valid_mask = ~np.isnan(readings[t])
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            estimates[t] = np.nan
            continue
        if len(valid_indices) == 1:
            estimates[t] = readings[t, valid_indices[0]]
            continue

        # Build mass functions
        masses = []
        for idx in valid_indices:
            op = _sensor_to_opinion(
                readings[t, idx], sigmas[idx], value_range, base_rate)
            masses.append({
                "b": op.belief, "d": op.disbelief, "u": op.uncertainty
            })

        # DS fuse
        fused = masses[0].copy()
        for m in masses[1:]:
            K = compute_conflict_K(fused, m)
            if K >= 0.999:
                continue  # skip total conflict
            fused = dempster_combine(fused, m)

        # Weight by projected probability (b + base_rate * u)
        weights = []
        for m in masses:
            weights.append(m["b"] + base_rate * m["u"])
        weights = np.array(weights)
        w_sum = np.sum(weights)
        if w_sum > 0:
            estimates[t] = np.sum(
                weights * readings[t, valid_indices]) / w_sum
        else:
            estimates[t] = np.mean(readings[t, valid_indices])

    return estimates


def sl_fuse_sensors_with_decay(
    readings: np.ndarray,
    sigmas: np.ndarray,
    staleness_hours: np.ndarray,
    value_range: Tuple[float, float],
    half_life_hours: float = 2.0,
    base_rate: float = 0.5,
) -> np.ndarray:
    """SL fusion with temporal decay on stale readings.

    Like sl_fuse_sensors but applies decay_opinion() to sensors
    whose readings are stale (staleness > 0).

    Args:
        readings: (n_steps, n_sensors) array.
        sigmas: (n_sensors,) noise std devs.
        staleness_hours: (n_sensors,) hours since last fresh reading.
            0 = current reading, >0 = stale.
        value_range: (min, max) of plausible values.
        half_life_hours: Decay half-life.
    """
    if readings.ndim == 1:
        readings = readings.reshape(1, -1)

    n_steps, n_sensors = readings.shape
    estimates = np.zeros(n_steps)

    for t in range(n_steps):
        valid_mask = ~np.isnan(readings[t])
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            estimates[t] = np.nan
            continue
        if len(valid_indices) == 1:
            estimates[t] = readings[t, valid_indices[0]]
            continue

        opinions = []
        for idx in valid_indices:
            op = _sensor_to_opinion(
                readings[t, idx], sigmas[idx], value_range, base_rate)

            # Apply decay if stale
            stale = staleness_hours[idx] if idx < len(staleness_hours) else 0
            if stale > 0:
                op = decay_opinion(op, elapsed=stale,
                                    half_life=half_life_hours)
            opinions.append(op)

        # Fuse
        fused = opinions[0]
        for op in opinions[1:]:
            fused = cumulative_fuse(fused, op)

        # Weight by evidence strength (1 - uncertainty)
        weights = np.array([1.0 - op.uncertainty for op in opinions])
        w_sum = np.sum(weights)
        if w_sum > 0:
            estimates[t] = np.sum(
                weights * readings[t, valid_indices]) / w_sum
        else:
            estimates[t] = np.mean(readings[t, valid_indices])

    return estimates


# =====================================================================
# 5. Evaluation Metrics
# =====================================================================

def compute_rmse(
    pred: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Root Mean Squared Error, ignoring NaN positions."""
    valid = ~(np.isnan(pred) | np.isnan(truth))
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean((pred[valid] - truth[valid]) ** 2)))


def compute_mae(
    pred: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Mean Absolute Error, ignoring NaN positions."""
    valid = ~(np.isnan(pred) | np.isnan(truth))
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(pred[valid] - truth[valid])))


# =====================================================================
# 6. Leave-One-Out Evaluation
# =====================================================================

def leave_one_out_eval(
    aligned: Dict[str, Any],
    sigmas: Dict[int, float],
    value_range: Tuple[float, float],
) -> Dict[str, Dict[str, float]]:
    """Leave-one-out sensor fusion evaluation.

    For each sensor, hold it out as ground truth and fuse the
    remaining sensors. Report RMSE and MAE per fusion method.

    Args:
        aligned: Output of time_align_readings. Has 'timestamps'
            and sensor_id -> values arrays.
        sigmas: Dict mapping sensor_id -> noise std dev.
        value_range: (min, max) for SL opinion construction.

    Returns:
        Dict keyed by method name, each with 'rmse' and 'mae'.
    """
    sensor_ids = [k for k in aligned if k != "timestamps"]
    n_steps = len(aligned["timestamps"])

    # Accumulate errors across all held-out sensors
    method_errors: Dict[str, List[float]] = {
        "scalar_avg": [], "weighted_avg": [], "kalman": [],
        "sl_fuse": [], "sl_fuse_decay": [], "ds_fuse": [],
    }

    for held_out in sensor_ids:
        others = [s for s in sensor_ids if s != held_out]
        if len(others) < 2:
            continue

        truth = aligned[held_out]
        sigma_arr = np.array([sigmas[s] for s in others])

        # Build readings matrix (n_steps, n_others)
        readings = np.column_stack([aligned[s] for s in others])

        # Valid time steps: held-out sensor has a reading
        valid_t = ~np.isnan(truth)
        if np.sum(valid_t) < 10:
            continue

        # Scalar average
        sa_pred = np.array([scalar_average(readings[t]) for t in range(n_steps)])
        method_errors["scalar_avg"].extend(
            (sa_pred[valid_t] - truth[valid_t]).tolist())

        # Weighted average
        wa_pred = np.array([
            weighted_average(readings[t], sigma_arr)
            for t in range(n_steps)])
        method_errors["weighted_avg"].extend(
            (wa_pred[valid_t] - truth[valid_t]).tolist())

        # Kalman
        k_pred = kalman_fuse_1d(readings, sigma_arr)
        method_errors["kalman"].extend(
            (k_pred[valid_t] - truth[valid_t]).tolist())

        # SL fusion
        sl_pred = sl_fuse_sensors(readings, sigma_arr, value_range)
        method_errors["sl_fuse"].extend(
            (sl_pred[valid_t] - truth[valid_t]).tolist())

        # SL fusion + decay (no staleness in LOO — use 0)
        staleness = np.zeros(len(others))
        sl_d_pred = sl_fuse_sensors_with_decay(
            readings, sigma_arr, staleness, value_range)
        method_errors["sl_fuse_decay"].extend(
            (sl_d_pred[valid_t] - truth[valid_t]).tolist())

        # DS fusion
        ds_pred = ds_fuse_sensors(readings, sigma_arr, value_range)
        method_errors["ds_fuse"].extend(
            (ds_pred[valid_t] - truth[valid_t]).tolist())

    # Compute RMSE and MAE per method
    results = {}
    for method, errors in method_errors.items():
        errors_arr = np.array(errors)
        # Filter out NaN
        valid = ~np.isnan(errors_arr)
        if np.any(valid):
            results[method] = {
                "rmse": float(np.sqrt(np.mean(errors_arr[valid] ** 2))),
                "mae": float(np.mean(np.abs(errors_arr[valid]))),
                "n_points": int(np.sum(valid)),
            }
        else:
            results[method] = {"rmse": float("nan"), "mae": float("nan"),
                               "n_points": 0}

    return results
