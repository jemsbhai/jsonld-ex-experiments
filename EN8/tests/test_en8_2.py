"""RED-phase tests for EN8.2 — IoT Sensor Pipeline.

Tests for:
  1. Intel Lab data loading and validation
  2. Time-alignment of asynchronous sensor readings
  3. Sensor fusion methods (6 conditions)
  4. SSN/SOSA round-trip gap analysis
  5. Code complexity comparison
"""
from __future__ import annotations

import math
import pytest
import numpy as np

from en8_2_core import (
    # Data loading
    load_intel_lab_data,
    load_mote_locations,
    select_sensor_cluster,
    time_align_readings,
    # Fusion baselines
    scalar_average,
    weighted_average,
    kalman_fuse_1d,
    gp_fuse_1d,
    # DS/SL fusion
    sl_fuse_sensors,
    ds_fuse_sensors,
    sl_fuse_sensors_with_decay,
    # Evaluation
    compute_rmse,
    compute_mae,
    leave_one_out_eval,
)


# ====================================================================
# Section 1: Intel Lab Data Loading
# ====================================================================

class TestDataLoading:
    """Verify Intel Lab data loads correctly."""

    def test_load_returns_records(self):
        """Data file loads and returns non-empty list of records."""
        data = load_intel_lab_data()
        assert len(data) > 100_000, f"Expected >100K records, got {len(data)}"

    def test_record_schema(self):
        """Each record has required fields."""
        data = load_intel_lab_data()
        rec = data[0]
        required = ["timestamp", "epoch", "moteid",
                     "temperature", "humidity", "light", "voltage"]
        for field in required:
            assert field in rec, f"Missing field: {field}"

    def test_temperature_range(self):
        """Temperature values are physically plausible after outlier removal.

        FINDING: Intel Lab data contains ~18% sensor failures (readings
        at 122°C, -38°C etc). This is expected for real IoT data and
        motivates outlier filtering in the pipeline.
        """
        data = load_intel_lab_data()
        temps = [r["temperature"] for r in data
                 if r["temperature"] is not None
                 and not math.isnan(r["temperature"])]
        # Raw data has ~18% outliers — this IS the real-world characteristic
        reasonable = [t for t in temps if 10 < t < 45]
        pct_reasonable = len(reasonable) / len(temps)
        # At least 75% should be in plausible range (actual: ~82%)
        assert pct_reasonable > 0.75, \
            f"Only {pct_reasonable:.1%} of temps in [10, 45]°C"
        # Verify outliers exist (confirms we need filtering)
        assert pct_reasonable < 0.95, \
            "Expected some sensor failures in real IoT data"

    def test_sensor_ids(self):
        """Sensor IDs include expected range (1-54 plus a few extras)."""
        data = load_intel_lab_data()
        ids = set(r["moteid"] for r in data)
        assert len(ids) >= 40, f"Expected >=40 unique sensors, got {len(ids)}"
        # Core sensors 1-54 present, plus extras (55, 56, 58)
        assert min(ids) >= 1
        # Verify most core sensors present
        core = set(range(1, 55))
        overlap = ids & core
        assert len(overlap) >= 40

    def test_mote_locations(self):
        """Mote locations load with x, y coordinates."""
        locs = load_mote_locations()
        assert len(locs) == 54
        assert 1 in locs
        assert isinstance(locs[1], tuple)
        assert len(locs[1]) == 2  # (x, y)


class TestSensorClustering:
    """Verify sensor cluster selection."""

    def test_select_cluster(self):
        """Selecting a cluster returns only those sensors' data.

        NOTE: Sensor 5 has only 35 readings (effectively dead).
        We use sensors [4, 6, 7, 9] instead.
        """
        data = load_intel_lab_data()
        cluster = select_sensor_cluster(data, [4, 6, 7, 9])
        sensor_ids = set(r["moteid"] for r in cluster)
        assert sensor_ids == {4, 6, 7, 9}
        assert len(cluster) > 1000  # substantial data


class TestTimeAlignment:
    """Verify time alignment of asynchronous readings."""

    def test_align_produces_grid(self):
        """Aligned data has uniform time steps."""
        data = load_intel_lab_data()
        cluster = select_sensor_cluster(data, [4, 6, 7, 9])
        aligned = time_align_readings(cluster, [4, 6, 7, 9],
                                       interval_seconds=300)
        assert isinstance(aligned, dict)
        assert "timestamps" in aligned
        assert 4 in aligned and 6 in aligned
        n = len(aligned["timestamps"])
        assert n > 100
        for sid in [4, 6, 7, 9]:
            assert len(aligned[sid]) == n

    def test_align_handles_gaps(self):
        """Aligned data uses NaN for missing readings."""
        data = load_intel_lab_data()
        cluster = select_sensor_cluster(data, [4, 6, 7, 9])
        aligned = time_align_readings(cluster, [4, 6, 7, 9],
                                       interval_seconds=300)
        for sid in [4, 6, 7, 9]:
            n_nan = np.sum(np.isnan(aligned[sid]))
            n_total = len(aligned[sid])
            assert n_nan < n_total * 0.8, \
                f"Sensor {sid}: {n_nan}/{n_total} NaN ({n_nan/n_total:.1%})"


# ====================================================================
# Section 2: Fusion Methods — Correctness
# ====================================================================

class TestScalarFusion:
    """Verify scalar averaging and weighted averaging."""

    def test_scalar_average_basic(self):
        """Equal-weight average of 3 sensors."""
        readings = np.array([20.0, 21.0, 22.0])
        assert scalar_average(readings) == pytest.approx(21.0)

    def test_scalar_average_with_nan(self):
        """NaN sensors are excluded from average."""
        readings = np.array([20.0, np.nan, 22.0])
        assert scalar_average(readings) == pytest.approx(21.0)

    def test_scalar_average_all_nan(self):
        """All NaN returns NaN."""
        readings = np.array([np.nan, np.nan])
        assert np.isnan(scalar_average(readings))

    def test_weighted_average(self):
        """Inverse-variance weighting."""
        readings = np.array([20.0, 22.0])
        sigmas = np.array([0.1, 1.0])  # first sensor much more precise
        result = weighted_average(readings, sigmas)
        # Should be much closer to 20.0
        assert 20.0 < result < 20.2

    def test_weighted_average_equal_weights(self):
        """Equal sigmas → equal weights → same as scalar average."""
        readings = np.array([20.0, 22.0])
        sigmas = np.array([1.0, 1.0])
        assert weighted_average(readings, sigmas) == pytest.approx(21.0)


class TestKalmanFusion:
    """Verify 1D Kalman filter for sensor fusion."""

    def test_kalman_reduces_noise(self):
        """Kalman estimate has lower variance than individual sensors."""
        rng = np.random.default_rng(42)
        true_signal = 20.0 * np.ones(100)
        noise1 = rng.normal(0, 1.0, 100)
        noise2 = rng.normal(0, 0.5, 100)
        readings = np.column_stack([true_signal + noise1,
                                     true_signal + noise2])
        sigmas = np.array([1.0, 0.5])
        estimates = kalman_fuse_1d(readings, sigmas)

        rmse_s1 = np.sqrt(np.mean(noise1**2))
        rmse_kalman = np.sqrt(np.mean((estimates - true_signal)**2))
        assert rmse_kalman < rmse_s1, \
            f"Kalman RMSE {rmse_kalman:.3f} should be < sensor1 RMSE {rmse_s1:.3f}"

    def test_kalman_handles_nan(self):
        """Kalman uses predict-only step when sensor is missing."""
        rng = np.random.default_rng(42)
        true_signal = np.linspace(20, 22, 50)
        readings = np.column_stack([
            true_signal + rng.normal(0, 0.5, 50),
            true_signal + rng.normal(0, 0.5, 50),
        ])
        # Drop 10 readings from sensor 2
        readings[20:30, 1] = np.nan
        sigmas = np.array([0.5, 0.5])
        estimates = kalman_fuse_1d(readings, sigmas)
        # Should still produce valid estimates
        assert not np.any(np.isnan(estimates))

    def test_kalman_output_length(self):
        """Output length matches input."""
        readings = np.ones((50, 3))
        sigmas = np.array([1.0, 1.0, 1.0])
        estimates = kalman_fuse_1d(readings, sigmas)
        assert len(estimates) == 50


class TestGPFusion:
    """Verify Gaussian Process regression for sensor fusion."""

    def test_gp_produces_estimates(self):
        """GP regression produces valid output."""
        rng = np.random.default_rng(42)
        n = 50
        true_signal = 20 + np.sin(np.linspace(0, 2*np.pi, n))
        readings = np.column_stack([
            true_signal + rng.normal(0, 0.3, n),
            true_signal + rng.normal(0, 0.5, n),
        ])
        sigmas = np.array([0.3, 0.5])
        estimates, uncertainties = gp_fuse_1d(readings, sigmas)
        assert len(estimates) == n
        assert len(uncertainties) == n
        assert not np.any(np.isnan(estimates))
        # Uncertainty should be positive
        assert np.all(uncertainties > 0)

    def test_gp_reduces_noise(self):
        """GP estimate closer to truth than individual sensors."""
        rng = np.random.default_rng(42)
        n = 100
        true_signal = 20 + 2 * np.sin(np.linspace(0, 4*np.pi, n))
        readings = np.column_stack([
            true_signal + rng.normal(0, 1.0, n),
            true_signal + rng.normal(0, 0.5, n),
        ])
        sigmas = np.array([1.0, 0.5])
        estimates, _ = gp_fuse_1d(readings, sigmas)

        rmse_s1 = np.sqrt(np.mean((readings[:, 0] - true_signal)**2))
        rmse_gp = np.sqrt(np.mean((estimates - true_signal)**2))
        assert rmse_gp < rmse_s1


class TestSLDSFusion:
    """Verify SL and DS fusion for sensor state estimation."""

    def test_sl_fuse_basic(self):
        """SL fusion produces valid estimates."""
        readings = np.array([[20.1, 20.3], [19.9, 20.0], [20.5, 20.2]])
        sigmas = np.array([0.3, 0.5])
        estimates = sl_fuse_sensors(readings, sigmas, value_range=(15, 35))
        assert len(estimates) == 3
        assert not np.any(np.isnan(estimates))
        # Estimates should be within the range of readings
        for i, est in enumerate(estimates):
            assert min(readings[i]) - 1 <= est <= max(readings[i]) + 1

    def test_ds_fuse_basic(self):
        """DS fusion produces valid estimates."""
        readings = np.array([[20.1, 20.3], [19.9, 20.0], [20.5, 20.2]])
        sigmas = np.array([0.3, 0.5])
        estimates = ds_fuse_sensors(readings, sigmas, value_range=(15, 35))
        assert len(estimates) == 3
        assert not np.any(np.isnan(estimates))

    def test_sl_decay_penalizes_stale(self):
        """SL+decay gives less weight to stale readings."""
        # Sensor 1: continuous, Sensor 2: stale (last reading 6h ago)
        readings = np.array([[20.0, 21.0]])  # current vs stale
        sigmas = np.array([0.5, 0.5])
        staleness_hours = np.array([0.0, 6.0])

        est_no_decay = sl_fuse_sensors(readings, sigmas, value_range=(15, 35))
        est_decay = sl_fuse_sensors_with_decay(
            readings, sigmas, staleness_hours, value_range=(15, 35),
            half_life_hours=2.0)

        # With decay, stale sensor 2 should contribute less
        # Result should be closer to sensor 1 (20.0) than without decay
        assert abs(est_decay[0] - 20.0) < abs(est_no_decay[0] - 20.0)


# ====================================================================
# Section 3: Evaluation Metrics
# ====================================================================

class TestMetrics:
    """Verify RMSE and MAE computation."""

    def test_rmse(self):
        pred = np.array([1.0, 2.0, 3.0])
        truth = np.array([1.1, 1.9, 3.2])
        rmse = compute_rmse(pred, truth)
        expected = np.sqrt(np.mean([0.01, 0.01, 0.04]))
        assert rmse == pytest.approx(expected, abs=1e-10)

    def test_mae(self):
        pred = np.array([1.0, 2.0, 3.0])
        truth = np.array([1.1, 1.9, 3.2])
        mae = compute_mae(pred, truth)
        expected = np.mean([0.1, 0.1, 0.2])
        assert mae == pytest.approx(expected, abs=1e-10)

    def test_rmse_handles_nan(self):
        """RMSE ignores NaN positions."""
        pred = np.array([1.0, np.nan, 3.0])
        truth = np.array([1.1, 2.0, 3.2])
        rmse = compute_rmse(pred, truth)
        expected = np.sqrt(np.mean([0.01, 0.04]))
        assert rmse == pytest.approx(expected, abs=1e-10)


class TestLeaveOneOut:
    """Verify leave-one-out evaluation harness."""

    def test_loo_returns_all_methods(self):
        """LOO eval returns results for all fusion methods."""
        # Small synthetic test
        rng = np.random.default_rng(42)
        n = 30
        true_val = 20.0
        aligned = {
            "timestamps": np.arange(n, dtype=float),
            1: true_val + rng.normal(0, 0.5, n),
            2: true_val + rng.normal(0, 0.3, n),
            3: true_val + rng.normal(0, 1.0, n),
        }
        sigmas = {1: 0.5, 2: 0.3, 3: 1.0}

        results = leave_one_out_eval(aligned, sigmas, value_range=(15, 35))

        expected_methods = ["scalar_avg", "weighted_avg", "kalman",
                            "sl_fuse", "sl_fuse_decay", "ds_fuse"]
        for method in expected_methods:
            assert method in results, f"Missing method: {method}"
            assert "rmse" in results[method]
            assert results[method]["rmse"] >= 0
