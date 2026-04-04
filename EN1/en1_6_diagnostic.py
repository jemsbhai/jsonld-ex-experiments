"""EN1.6 Diagnostic -- understand actual behavior before writing tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from en1_6_core import (
    generate_signal, generate_sensor_readings, SENSOR_SPECS,
    fuse_scalar_weighted, fuse_kalman, fuse_sl, fuse_sl_temporal,
    compute_fusion_metrics, SCENARIOS,
)

print("=" * 70)
print("  EN1.6 DIAGNOSTIC: Understanding actual method behavior")
print("=" * 70)

# Run all scenarios with enough reps for stable estimates
for scenario in SCENARIOS:
    signal = generate_signal(scenario, n_steps=200, seed=42)
    readings = generate_sensor_readings(signal, SENSOR_SPECS, seed=42)
    true_states = signal["true_state"]

    # All 4 methods
    preds_sw = fuse_scalar_weighted(readings, SENSOR_SPECS)
    preds_k = fuse_kalman(readings, SENSOR_SPECS)
    result_sl = fuse_sl(readings, SENSOR_SPECS)
    result_slt = fuse_sl_temporal(readings, SENSOR_SPECS)

    m_sw = compute_fusion_metrics(preds_sw, true_states)
    m_k = compute_fusion_metrics(preds_k, true_states)
    m_sl = compute_fusion_metrics(
        result_sl["predictions"], true_states,
        uncertainties=result_sl["uncertainties"],
    )
    m_slt = compute_fusion_metrics(
        result_slt["predictions"], true_states,
        uncertainties=result_slt["uncertainties"],
    )

    print(f"\n  Scenario: {scenario}")
    print(f"  {'Method':<20} {'MAE':>8} {'F1':>8} {'ECE':>8} {'Acc':>8}")
    print(f"  {'-'*56}")
    print(f"  {'scalar_weighted':<20} {m_sw['mae']:>8.4f} {m_sw['f1']:>8.4f} {m_sw['ece']:>8.4f} {m_sw['accuracy']:>8.4f}")
    print(f"  {'kalman':<20} {m_k['mae']:>8.4f} {m_k['f1']:>8.4f} {m_k['ece']:>8.4f} {m_k['accuracy']:>8.4f}")
    print(f"  {'sl_fusion':<20} {m_sl['mae']:>8.4f} {m_sl['f1']:>8.4f} {m_sl['ece']:>8.4f} {m_sl['accuracy']:>8.4f}")
    print(f"  {'sl_temporal':<20} {m_slt['mae']:>8.4f} {m_slt['f1']:>8.4f} {m_slt['ece']:>8.4f} {m_slt['accuracy']:>8.4f}")

    # Investigate SL uncertainty during gaps
    avail_s3 = readings[2]["available"]
    sl_u = result_sl["uncertainties"]
    u_with_s3 = [sl_u[i] for i in range(len(sl_u)) if avail_s3[i]]
    u_without_s3 = [sl_u[i] for i in range(len(sl_u)) if not avail_s3[i]]
    if u_with_s3 and u_without_s3:
        print(f"  SL uncertainty: with_s3={np.mean(u_with_s3):.4f}  "
              f"without_s3={np.mean(u_without_s3):.4f}  "
              f"diff={np.mean(u_without_s3) - np.mean(u_with_s3):+.4f}")

# Deep dive: look at the first 30 timesteps of gradual_rise
print(f"\n{'=' * 70}")
print(f"  DEEP DIVE: gradual_rise first 30 steps")
print(f"{'=' * 70}")
signal = generate_signal("gradual_rise", n_steps=200, seed=42)
readings = generate_sensor_readings(signal, SENSOR_SPECS, seed=42)
result_sl = fuse_sl(readings, SENSOR_SPECS)

print(f"  {'t':>3} {'true':>4} {'val':>6} {'s1':>3} {'s2':>3} {'s3':>3} "
      f"{'s3av':>4} {'sl_p':>6} {'sl_u':>6} {'sw_p':>6}")
preds_sw = fuse_scalar_weighted(readings, SENSOR_SPECS)
for t in range(30):
    s1 = readings[0]["observations"][t] if readings[0]["available"][t] else "-"
    s2 = readings[1]["observations"][t] if readings[1]["available"][t] else "-"
    s3 = readings[2]["observations"][t] if readings[2]["available"][t] else "-"
    s3a = "Y" if readings[2]["available"][t] else "N"
    print(f"  {t:>3} {signal['true_state'][t]:>4} {signal['value'][t]:>6.3f} "
          f"{str(s1):>3} {str(s2):>3} {str(s3):>3} {s3a:>4} "
          f"{result_sl['predictions'][t]:>6.3f} {result_sl['uncertainties'][t]:>6.3f} "
          f"{preds_sw[t]:>6.3f}")
