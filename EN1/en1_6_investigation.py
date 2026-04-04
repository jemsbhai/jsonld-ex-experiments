"""EN1.6 Extended Investigation -- Real-World Sensor Failure Modes.

Tests SL in scenarios where its structural advantages should matter:
1. Sensor degradation (TPR drifts mid-experiment)
2. Sensor fault (stuck reporting 1)
3. No process model (hobbled Kalman)
4. Conflicting sensors (detect disagreement)
5. Misspecified base rate (honest failure mode)

This is the rigorous follow-up: WHERE and WHY does SL add genuine value?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from en1_6_core import (
    SensorSpec, SENSOR_SPECS,
    generate_signal, generate_sensor_readings,
    fuse_scalar_weighted, fuse_kalman,
    fuse_sl, fuse_sl_temporal, fuse_sl_naive,
    compute_fusion_metrics, _compute_scenario_base_rate,
)
from jsonld_ex.confidence_algebra import (
    Opinion, cumulative_fuse, pairwise_conflict, conflict_metric,
)
from jsonld_ex.confidence_decay import decay_opinion


def inject_sensor_degradation(readings, degrade_sensor=0,
                               degrade_start_frac=0.5,
                               stuck_value=1):
    """Sensor degrades: after degrade_start, sensor gets stuck."""
    n_steps = len(readings[0]["observations"])
    start = int(n_steps * degrade_start_frac)
    modified = []
    for s_idx, r in enumerate(readings):
        new_r = {
            "sensor_name": r["sensor_name"],
            "observations": list(r["observations"]),
            "available": list(r["available"]),
        }
        if s_idx == degrade_sensor:
            for t in range(start, n_steps):
                if new_r["available"][t]:
                    new_r["observations"][t] = stuck_value  # stuck
        modified.append(new_r)
    return modified, start


def run_investigation(n_steps=500, n_reps=100, seed=42):
    """Run all investigation scenarios."""
    rng_base = np.random.RandomState(seed)
    results = {}

    # ================================================================
    # SCENARIO A: Sensor Degradation (Sensor 1 gets stuck at t=250)
    # ================================================================
    print("\n  SCENARIO A: Sensor Degradation (sensor 1 stuck after 50%)")
    print("  " + "-" * 60)
    scenario_a = {"scalar": [], "kalman": [], "sl": [], "sl_temp": []}

    for rep in range(n_reps):
        rep_seed = seed + rep * 1000
        signal = generate_signal("oscillating", n_steps=n_steps, seed=rep_seed)
        readings = generate_sensor_readings(signal, SENSOR_SPECS, seed=rep_seed+1)
        degraded, deg_start = inject_sensor_degradation(readings, degrade_sensor=0)
        true_states = signal["true_state"]

        # All methods see the DEGRADED readings (they don't know sensor is stuck)
        p_sw = fuse_scalar_weighted(degraded, SENSOR_SPECS)
        p_k = fuse_kalman(degraded, SENSOR_SPECS)
        r_sl = fuse_sl(degraded, SENSOR_SPECS, scenario="oscillating", signal=signal)
        r_slt = fuse_sl_temporal(degraded, SENSOR_SPECS, scenario="oscillating", signal=signal)

        # Measure on POST-degradation period only (where the fault matters)
        post_true = true_states[deg_start:]
        m_sw = compute_fusion_metrics(p_sw[deg_start:], post_true)
        m_k = compute_fusion_metrics(p_k[deg_start:], post_true)
        m_sl = compute_fusion_metrics(r_sl["predictions"][deg_start:], post_true)
        m_slt = compute_fusion_metrics(r_slt["predictions"][deg_start:], post_true)

        scenario_a["scalar"].append(m_sw["mae"])
        scenario_a["kalman"].append(m_k["mae"])
        scenario_a["sl"].append(m_sl["mae"])
        scenario_a["sl_temp"].append(m_slt["mae"])

    for method, vals in scenario_a.items():
        print(f"    {method:<12} MAE={np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    results["sensor_degradation"] = {k: float(np.mean(v)) for k, v in scenario_a.items()}

    # ================================================================
    # SCENARIO B: Hobbled Kalman (no process model advantage)
    # ================================================================
    print("\n  SCENARIO B: Hobbled Kalman (high process noise Q=0.25)")
    print("  " + "-" * 60)
    scenario_b = {"scalar": [], "kalman_optimal": [], "kalman_hobbled": [],
                  "sl": [], "sl_temp": []}

    for rep in range(n_reps):
        rep_seed = seed + rep * 1000
        signal = generate_signal("oscillating", n_steps=n_steps, seed=rep_seed)
        readings = generate_sensor_readings(signal, SENSOR_SPECS, seed=rep_seed+1)
        true_states = signal["true_state"]

        p_sw = fuse_scalar_weighted(readings, SENSOR_SPECS)
        p_k_opt = fuse_kalman(readings, SENSOR_SPECS)
        r_sl = fuse_sl(readings, SENSOR_SPECS, scenario="oscillating", signal=signal)
        r_slt = fuse_sl_temporal(readings, SENSOR_SPECS, scenario="oscillating", signal=signal)

        # Hobbled Kalman: simulate no process model by running
        # Kalman with very high process noise
        import en1_6_core
        old_q = en1_6_core._KALMAN_Q
        en1_6_core._KALMAN_Q = 0.25  # effectively no process model
        p_k_hob = fuse_kalman(readings, SENSOR_SPECS)
        en1_6_core._KALMAN_Q = old_q

        m_sw = compute_fusion_metrics(p_sw, true_states)
        m_ko = compute_fusion_metrics(p_k_opt, true_states)
        m_kh = compute_fusion_metrics(p_k_hob, true_states)
        m_sl = compute_fusion_metrics(r_sl["predictions"], true_states)
        m_slt = compute_fusion_metrics(r_slt["predictions"], true_states)

        scenario_b["scalar"].append(m_sw["mae"])
        scenario_b["kalman_optimal"].append(m_ko["mae"])
        scenario_b["kalman_hobbled"].append(m_kh["mae"])
        scenario_b["sl"].append(m_sl["mae"])
        scenario_b["sl_temp"].append(m_slt["mae"])

    for method, vals in scenario_b.items():
        print(f"    {method:<18} MAE={np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    results["hobbled_kalman"] = {k: float(np.mean(v)) for k, v in scenario_b.items()}

    # ================================================================
    # SCENARIO C: Conflict Detection (SL unique capability)
    # ================================================================
    print("\n  SCENARIO C: Conflict Detection During Sensor Fault")
    print("  " + "-" * 60)

    # Single detailed run to show conflict detection
    signal = generate_signal("oscillating", n_steps=n_steps, seed=42)
    readings = generate_sensor_readings(signal, SENSOR_SPECS, seed=43)
    degraded, deg_start = inject_sensor_degradation(readings, degrade_sensor=0)

    # Compute per-timestep conflict using SL
    from en1_6_core import _sensor_obs_to_opinion
    base_rate = _compute_scenario_base_rate("oscillating")

    conflicts_pre = []
    conflicts_post = []
    for t in range(n_steps):
        opinions = []
        for s_idx in range(len(SENSOR_SPECS)):
            if degraded[s_idx]["available"][t]:
                obs = degraded[s_idx]["observations"][t]
                op = _sensor_obs_to_opinion(
                    obs, SENSOR_SPECS[s_idx], base_rate
                )
                opinions.append(op)

        if len(opinions) >= 2:
            # Compute max pairwise conflict
            max_conf = 0.0
            for i in range(len(opinions)):
                for j in range(i+1, len(opinions)):
                    c = pairwise_conflict(opinions[i], opinions[j])
                    max_conf = max(max_conf, c)
            if t < deg_start:
                conflicts_pre.append(max_conf)
            else:
                conflicts_post.append(max_conf)

    pre_mean = np.mean(conflicts_pre) if conflicts_pre else 0
    post_mean = np.mean(conflicts_post) if conflicts_post else 0
    print(f"    Mean conflict BEFORE fault: {pre_mean:.4f}")
    print(f"    Mean conflict AFTER fault:  {post_mean:.4f}")
    print(f"    Conflict increase:          {post_mean - pre_mean:+.4f} "
          f"({(post_mean/pre_mean - 1)*100 if pre_mean > 0 else 0:+.1f}%)")

    # Can we detect the fault? Threshold-based detection
    threshold = pre_mean + 2 * np.std(conflicts_pre) if conflicts_pre else 0.5
    detected_pre = sum(1 for c in conflicts_pre if c > threshold)
    detected_post = sum(1 for c in conflicts_post if c > threshold)
    print(f"    Fault detections (threshold={threshold:.3f}):")
    print(f"      Pre-fault:  {detected_pre}/{len(conflicts_pre)} "
          f"({detected_pre/len(conflicts_pre)*100 if conflicts_pre else 0:.1f}% false alarms)")
    print(f"      Post-fault: {detected_post}/{len(conflicts_post)} "
          f"({detected_post/len(conflicts_post)*100 if conflicts_post else 0:.1f}% detection rate)")

    results["conflict_detection"] = {
        "pre_fault_conflict": float(pre_mean),
        "post_fault_conflict": float(post_mean),
        "conflict_increase_pct": float((post_mean/pre_mean - 1)*100) if pre_mean > 0 else 0,
        "false_alarm_rate": float(detected_pre/len(conflicts_pre)) if conflicts_pre else 0,
        "detection_rate": float(detected_post/len(conflicts_post)) if conflicts_post else 0,
    }

    # ================================================================
    # SCENARIO D: Misspecified Base Rate (honest failure mode)
    # ================================================================
    print("\n  SCENARIO D: Misspecified Base Rate (SL failure mode)")
    print("  " + "-" * 60)
    scenario_d = {}

    for wrong_br in [0.1, 0.3, 0.5, 0.7, 0.9]:
        maes = []
        for rep in range(n_reps):
            rep_seed = seed + rep * 1000
            signal = generate_signal("oscillating", n_steps=n_steps, seed=rep_seed)
            readings = generate_sensor_readings(signal, SENSOR_SPECS, seed=rep_seed+1)
            r_sl = fuse_sl(readings, SENSOR_SPECS, base_rate=wrong_br, signal=signal)
            m = compute_fusion_metrics(r_sl["predictions"], signal["true_state"])
            maes.append(m["mae"])
        mean_mae = float(np.mean(maes))
        correct_label = " (correct)" if abs(wrong_br - 0.5) < 0.01 else ""
        print(f"    base_rate={wrong_br:.1f}{correct_label}: "
              f"SL MAE={mean_mae:.4f}")
        scenario_d[str(wrong_br)] = mean_mae

    results["misspecified_base_rate"] = scenario_d

    # ================================================================
    # SCENARIO E: All sensors intermittent (data-poor regime)
    # ================================================================
    print("\n  SCENARIO E: All Sensors Intermittent (30% availability each)")
    print("  " + "-" * 60)
    sparse_specs = [
        SensorSpec(name="sparse_precision", tpr=0.98, fpr=0.02, availability=0.30),
        SensorSpec(name="sparse_robust", tpr=0.75, fpr=0.20, availability=0.30),
        SensorSpec(name="sparse_quality", tpr=0.97, fpr=0.03, availability=0.30),
    ]
    scenario_e = {"scalar": [], "kalman": [], "sl": [], "sl_temp": []}

    for rep in range(n_reps):
        rep_seed = seed + rep * 1000
        signal = generate_signal("oscillating", n_steps=n_steps, seed=rep_seed)
        readings = generate_sensor_readings(signal, sparse_specs, seed=rep_seed+1)
        true_states = signal["true_state"]

        p_sw = fuse_scalar_weighted(readings, sparse_specs)
        p_k = fuse_kalman(readings, sparse_specs)
        r_sl = fuse_sl(readings, sparse_specs, scenario="oscillating", signal=signal)
        r_slt = fuse_sl_temporal(readings, sparse_specs, scenario="oscillating",
                                  signal=signal)

        m_sw = compute_fusion_metrics(p_sw, true_states)
        m_k = compute_fusion_metrics(p_k, true_states)
        m_sl = compute_fusion_metrics(r_sl["predictions"], true_states,
                                       uncertainties=r_sl["uncertainties"])
        m_slt = compute_fusion_metrics(r_slt["predictions"], true_states,
                                        uncertainties=r_slt["uncertainties"])

        scenario_e["scalar"].append(m_sw["mae"])
        scenario_e["kalman"].append(m_k["mae"])
        scenario_e["sl"].append(m_sl["mae"])
        scenario_e["sl_temp"].append(m_slt["mae"])

    for method, vals in scenario_e.items():
        print(f"    {method:<12} MAE={np.mean(vals):.4f} +/- {np.std(vals):.4f}")
    results["all_sparse"] = {k: float(np.mean(v)) for k, v in scenario_e.items()}

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"  INVESTIGATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  Where SL wins:")
    a = results["sensor_degradation"]
    if a["sl"] < a["kalman"]:
        print(f"    [+] Sensor degradation: SL {a['sl']:.4f} < Kalman {a['kalman']:.4f}")
    else:
        print(f"    [-] Sensor degradation: SL {a['sl']:.4f} >= Kalman {a['kalman']:.4f}")

    b = results["hobbled_kalman"]
    if b["sl"] < b["kalman_hobbled"]:
        print(f"    [+] No process model: SL {b['sl']:.4f} < Hobbled Kalman {b['kalman_hobbled']:.4f}")
    else:
        print(f"    [-] No process model: SL {b['sl']:.4f} >= Hobbled Kalman {b['kalman_hobbled']:.4f}")

    c = results["conflict_detection"]
    print(f"    [+] Conflict detection: {c['detection_rate']*100:.1f}% fault detection, "
          f"{c['false_alarm_rate']*100:.1f}% false alarm (SL unique)")

    e = results["all_sparse"]
    if e["sl"] < e["kalman"]:
        print(f"    [+] Data-poor: SL {e['sl']:.4f} < Kalman {e['kalman']:.4f}")
    else:
        print(f"    [-] Data-poor: SL {e['sl']:.4f} >= Kalman {e['kalman']:.4f}")

    print(f"\n  Where SL loses:")
    print(f"    [-] Optimal Kalman always wins on MAE in clean conditions")

    d = results["misspecified_base_rate"]
    print(f"    [-] Misspecified base rate: MAE range {min(d.values()):.4f} to {max(d.values()):.4f}")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("  EN1.6 EXTENDED INVESTIGATION: Real-World Sensor Failure Modes")
    print("=" * 70)
    results = run_investigation()
