#!/usr/bin/env python
"""EN8.2 -- IoT Sensor Pipeline: Full Runner.

Phases:
    A: Pipeline throughput benchmark (H8.2a)
    B: SSN/SOSA interoperability gap analysis (H8.2b)
    C: Transport efficiency -- CBOR-LD, MQTT, CoAP (H8.2c)
    D: Data quality detection on Intel Lab (H8.2d)
    E: Code complexity comparison (H8.2e)
    F: Real-world data characteristics report (H8.2f)

Usage:
    python experiments/EN8/run_en8_2.py --phase a
    python experiments/EN8/run_en8_2.py --phase all
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

_EN8_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_ROOT = _EN8_DIR.parent
_PKG_SRC = _EXPERIMENTS_ROOT.parent / "packages" / "python" / "src"
for p in [str(_EN8_DIR), str(_EXPERIMENTS_ROOT), str(_PKG_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from en8_2_core import (
    load_intel_lab_data, select_sensor_cluster, time_align_readings,
    scalar_average, weighted_average, kalman_fuse_1d,
)
from en8_2_pipeline import (
    annotate_sensor_reading, readings_to_ssn, ssn_round_trip,
    categorize_ssn_triples, validate_ssn_structure,
    encode_cbor_batch, encode_json_batch, mqtt_derive_batch, coap_derive_batch,
    compute_disagreement_signal, detect_drift, detect_outliers_by_annotation,
    pipeline_jsonldex, pipeline_rdflib, count_loc, count_ssn_uris,
)

RESULTS_DIR = _EN8_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SEED = 42
CLUSTERS = {"primary": [4, 6, 7, 9], "validation": [22, 23, 24, 25, 26]}


def filter_outliers(data, lo=10.0, hi=45.0):
    return [r for r in data if r["temperature"] is not None
            and lo <= r["temperature"] <= hi]


def run_phase_a(data):
    """Phase A: Pipeline Throughput (H8.2a)."""
    print("\n" + "=" * 60)
    print("Phase A: Pipeline Throughput Benchmark")
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    sample = [data[i] for i in rng.choice(len(data), min(10000, len(data)), replace=False)]
    import jsonld_ex as jx

    t0 = time.perf_counter()
    for r in sample:
        annotate_sensor_reading(value=r["temperature"], sensor_id=f"sensor-{r['moteid']}", unit="celsius", sigma=0.3)
    t_ann = time.perf_counter() - t0
    rate_ann = len(sample) / t_ann

    t0 = time.perf_counter()
    for r in sample[:2000]:
        a = annotate_sensor_reading(value=r["temperature"], sensor_id=f"sensor-{r['moteid']}", unit="celsius", sigma=0.3)
        readings_to_ssn(a)
    t_ssn = time.perf_counter() - t0
    rate_ssn = 2000 / t_ssn

    t0 = time.perf_counter()
    for r in sample[:1000]:
        a = annotate_sensor_reading(value=r["temperature"], sensor_id=f"sensor-{r['moteid']}", unit="celsius", sigma=0.3)
        ssn, _ = readings_to_ssn(a)
        jx.to_cbor(ssn)
    t_all = time.perf_counter() - t0
    rate_all = 1000 / t_all

    print(f"  annotate:          {rate_ann:,.0f}/s")
    print(f"  annotate+SSN:      {rate_ssn:,.0f}/s")
    print(f"  annotate+SSN+CBOR: {rate_all:,.0f}/s")
    return {"annotate": rate_ann, "annotate_ssn": rate_ssn, "full": rate_all}


def run_phase_b(data):
    """Phase B: SSN/SOSA Gap Analysis (H8.2b)."""
    print("\n" + "=" * 60)
    print("Phase B: SSN/SOSA Interoperability Gap Analysis")
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    sample = [data[i] for i in rng.choice(len(data), 500, replace=False)]

    check_fields = ["@value", "@unit", "@source", "@confidence", "@measurementUncertainty"]
    total_checked, total_preserved = 0, 0
    field_stats = {f: {"checked": 0, "preserved": 0} for f in check_fields}
    cat_totals = {"native": 0, "allied": 0, "extension": 0}
    struct_pass = 0
    json_sizes, ssn_sizes = [], []

    for r in sample:
        doc = annotate_sensor_reading(value=r["temperature"], sensor_id=f"sensor-{r['moteid']}", unit="celsius", sigma=0.3, calibrated_at="2024-01-15T10:00:00Z")
        orig, rec, _ = ssn_round_trip(doc)
        for f in check_fields:
            if f in orig:
                total_checked += 1
                field_stats[f]["checked"] += 1
                if f in rec:
                    match = False
                    if isinstance(orig[f], float):
                        match = abs(orig[f] - rec.get(f, -999)) < 1e-6
                    else:
                        match = orig[f] == rec.get(f)
                    if match:
                        total_preserved += 1
                        field_stats[f]["preserved"] += 1

        ssn_doc, _ = readings_to_ssn(doc)
        cats = categorize_ssn_triples(ssn_doc)
        for k in cat_totals:
            cat_totals[k] += cats.get(k, 0)
        v = validate_ssn_structure(ssn_doc)
        if all(v.values()):
            struct_pass += 1
        wrapper = {"@context": {"@vocab": "https://schema.org/"}, "@type": "Observation", "t": doc}
        json_sizes.append(len(json.dumps(wrapper).encode()))
        ssn_sizes.append(len(json.dumps(ssn_doc).encode()))

    fidelity = total_preserved / total_checked if total_checked else 0
    total_t = sum(cat_totals.values())
    overhead = np.mean(ssn_sizes) / np.mean(json_sizes) if json_sizes else 0

    print(f"  Fidelity: {fidelity:.1%} ({total_preserved}/{total_checked})")
    for f in check_fields:
        s = field_stats[f]
        p = s["preserved"] / s["checked"] if s["checked"] else 0
        print(f"    {f:30s}: {p:.1%}")
    print(f"  Triples: native={cat_totals['native']} allied={cat_totals['allied']} ext={cat_totals['extension']}")
    print(f"  Structure pass: {struct_pass}/{len(sample)}")
    print(f"  Byte overhead: {overhead:.2f}x")

    return {"fidelity": fidelity, "per_field": field_stats, "triples": cat_totals,
            "struct_pass": struct_pass, "byte_overhead": overhead, "n": len(sample)}


def run_phase_c(data):
    """Phase C: Transport Efficiency (H8.2c)."""
    print("\n" + "=" * 60)
    print("Phase C: Transport Efficiency")
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    sample = [data[i] for i in rng.choice(len(data), 1000, replace=False)]

    results = {}
    for label, extra_kw in [("minimal", {}), ("full", {"calibrated_at": "2024-01-15T10:00:00Z", "calibration_method": "NIST"})]:
        docs = [annotate_sensor_reading(value=r["temperature"], sensor_id=f"sensor-{r['moteid']}", unit="celsius", sigma=0.3, **extra_kw) for r in sample]
        cs = encode_cbor_batch(docs)
        js = encode_json_batch(docs)
        ratio = sum(cs) / sum(js)
        results[label] = {"json": sum(js), "cbor": sum(cs), "ratio": ratio}
        print(f"  {label:10s}: JSON={sum(js):,} CBOR={sum(cs):,} ratio={ratio:.1%}")

    mqtt = mqtt_derive_batch([annotate_sensor_reading(value=20.0, sensor_id="s1", unit="celsius", sigma=0.3)])
    coap = coap_derive_batch([annotate_sensor_reading(value=20.0, sensor_id="s1", unit="celsius", sigma=0.3)])
    print(f"  MQTT topic: {mqtt[0]['topic']}  QoS: {mqtt[0]['qos']}")
    print(f"  CoAP path: {coap[0].get('uri_path')}")
    results["mqtt"] = mqtt[0]
    results["coap"] = coap[0]
    return results


def run_phase_d(data):
    """Phase D: Data Quality Detection (H8.2d)."""
    print("\n" + "=" * 60)
    print("Phase D: Data Quality Detection")
    print("=" * 60)
    raw_data = load_intel_lab_data()  # unfiltered for outlier detection
    all_results = {}

    for cname, sids in CLUSTERS.items():
        print(f"\n  --- {cname} ({sids}) ---")
        cdata = select_sensor_cluster(data, sids)
        aligned = time_align_readings(cdata, sids, interval_seconds=300)
        n = len(aligned["timestamps"])
        readings = {f"s{sid}": aligned[sid] for sid in sids}

        disagr = compute_disagreement_signal(readings)
        drift = detect_drift(readings, n_chunks=10)

        # Ground-truth bias
        biases = {}
        sigmas_est = {}
        for sid in sids:
            others = [s for s in sids if s != sid]
            devs = [aligned[sid][t] - np.mean([aligned[s][t] for s in others if not np.isnan(aligned[s][t])])
                    for t in range(n) if not np.isnan(aligned[sid][t]) and sum(1 for s in others if not np.isnan(aligned[s][t])) >= 2]
            if devs:
                d = np.array(devs)
                biases[f"s{sid}"] = float(np.median(d))
                sigmas_est[f"s{sid}"] = max(0.05, float(np.median(np.abs(d - np.median(d))) * 1.4826))

        skeys = sorted(disagr.keys())
        ba = [abs(biases.get(k, 0)) for k in skeys]
        dv = [disagr[k] for k in skeys]
        corr = float(np.corrcoef(ba, dv)[0, 1]) if len(skeys) >= 3 else float("nan")

        print(f"  Bias-disagr corr: r={corr:.3f}")
        for k in skeys:
            print(f"    {k}: disagr={disagr[k]:.3f} drift={drift[k]:.3f} bias={biases.get(k,0):+.3f}")

        # Outlier detection on raw data
        raw_c = select_sensor_cluster(raw_data, sids)
        raw_al = time_align_readings(raw_c, sids, interval_seconds=300)
        outlier_res = {}
        for sid in sids:
            v = raw_al[sid]
            valid = ~np.isnan(v)
            is_out = valid & ((v < 10) | (v > 45))
            n_out = int(np.sum(is_out))
            if n_out == 0:
                continue
            sig = sigmas_est.get(f"s{sid}", 0.3)
            flagged = detect_outliers_by_annotation(v, sigma=sig, threshold_sigmas=5.0, neighbor_values={s: raw_al[s] for s in sids if s != sid})
            tp = int(np.sum(flagged & is_out))
            recall = tp / n_out
            fp = int(np.sum(flagged & ~is_out & valid))
            fpr = fp / int(np.sum(~is_out & valid)) if np.sum(~is_out & valid) > 0 else 0
            outlier_res[str(sid)] = {"n": n_out, "tp": tp, "recall": recall, "fpr": fpr}
            print(f"  Outlier sensor {sid}: {tp}/{n_out} ({recall:.1%} recall, {fpr:.1%} FPR)")

        all_results[cname] = {"disagr": disagr, "drift": drift, "biases": biases,
                              "corr": corr, "outliers": outlier_res}
    return all_results


def run_phase_e():
    """Phase E: Code Complexity (H8.2e)."""
    print("\n" + "=" * 60)
    print("Phase E: Code Complexity")
    print("=" * 60)
    jx_loc = count_loc(pipeline_jsonldex)
    rdf_loc = count_loc(pipeline_rdflib)
    jx_uris = count_ssn_uris(pipeline_jsonldex)
    rdf_uris = count_ssn_uris(pipeline_rdflib)
    sensors = [{"id": f"s-{i}", "sigma": 0.3} for i in range(5)]
    readings = [{"sensor_id": f"s-{i%5}", "value": 20.0+i*0.1, "unit": "celsius",
                 "timestamp": f"2024-03-01T{i//60:02d}:{i%60:02d}:00Z", "sigma": 0.3} for i in range(100)]
    t0 = time.perf_counter()
    for _ in range(10):
        pipeline_jsonldex(sensors, readings)
    jx_t = (time.perf_counter() - t0) / 10
    t0 = time.perf_counter()
    for _ in range(10):
        pipeline_rdflib(sensors, readings)
    rdf_t = (time.perf_counter() - t0) / 10

    ratio = jx_loc / rdf_loc if rdf_loc else 0
    print(f"  jsonld-ex: {jx_loc} LoC, {jx_uris} SSN URIs, {jx_t:.3f}s")
    print(f"  rdflib:    {rdf_loc} LoC, {rdf_uris} SSN URIs, {rdf_t:.3f}s")
    print(f"  LoC ratio: {ratio:.1%}")
    return {"jx_loc": jx_loc, "rdf_loc": rdf_loc, "jx_uris": jx_uris,
            "rdf_uris": rdf_uris, "ratio": ratio}


def run_phase_f(data):
    """Phase F: Data Characteristics Report (H8.2f)."""
    print("\n" + "=" * 60)
    print("Phase F: Real-World Data Characteristics")
    print("=" * 60)
    from scipy import stats as sp
    all_results = {}
    for cname, sids in CLUSTERS.items():
        print(f"\n  --- {cname} ({sids}) ---")
        cdata = select_sensor_cluster(data, sids)
        aligned = time_align_readings(cdata, sids, interval_seconds=300)
        n = len(aligned["timestamps"])
        biases = {}
        for sid in sids:
            others = [s for s in sids if s != sid]
            devs = [aligned[sid][t] - np.mean([aligned[s][t] for s in others if not np.isnan(aligned[s][t])])
                    for t in range(n) if not np.isnan(aligned[sid][t]) and sum(1 for s in others if not np.isnan(aligned[s][t])) >= 2]
            biases[sid] = float(np.median(devs)) if devs else 0.0
        cres = {}
        for sid in sids:
            others = [s for s in sids if s != sid]
            res = []
            for t in range(n):
                v = aligned[sid][t] - biases[sid]
                if np.isnan(v):
                    continue
                ov = [aligned[s][t] - biases[s] for s in others if not np.isnan(aligned[s][t])]
                if len(ov) >= 2:
                    res.append(v - np.mean(ov))
            r = np.array(res)
            if len(r) < 20:
                continue
            ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
            ac5 = float(np.corrcoef(r[:-5], r[5:])[0, 1]) if len(r) > 10 else 0
            kurt = float(sp.kurtosis(r))
            skw = float(sp.skew(r))
            sig = float(np.std(r))
            b3 = float(np.mean(np.abs(r) > 3*sig)) * 100
            chunk_b = []
            cs = max(1, n // 10)
            for c in range(10):
                cd = [aligned[sid][t] - np.mean([aligned[s][t] for s in others if not np.isnan(aligned[s][t])])
                      for t in range(c*cs, min((c+1)*cs, n))
                      if not np.isnan(aligned[sid][t]) and sum(1 for s in others if not np.isnan(aligned[s][t])) >= 2]
                if cd:
                    chunk_b.append(float(np.median(cd)))
            dr = max(chunk_b) - min(chunk_b) if len(chunk_b) >= 2 else 0
            msig = max(0.05, float(np.median(np.abs(r)) * 1.4826))
            cres[str(sid)] = {"ac1": ac1, "ac5": ac5, "kurt": kurt, "skew": skw,
                              "beyond3sig": b3, "drift": dr, "drift_sigma": dr/msig,
                              "bias": biases[sid]}
            print(f"    Sensor {sid}: ac1={ac1:.3f} kurt={kurt:.1f} drift/sig={dr/msig:.1f}x")
        all_results[cname] = cres
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["a","b","c","d","e","f","all"], default="all")
    args = parser.parse_args()

    print("=" * 60)
    print("EN8.2 -- IoT Sensor Pipeline: Full Evaluation")
    print("=" * 60)

    results = {"experiment": "EN8.2", "timestamp": datetime.now(timezone.utc).isoformat(), "seed": SEED}
    data = None
    if args.phase in ("a","b","c","d","f","all"):
        t0 = time.time()
        raw = load_intel_lab_data()
        data = filter_outliers(raw)
        results["n_raw"] = len(raw)
        results["n_filtered"] = len(data)
        print(f"  Data: {len(raw):,} raw, {len(data):,} filtered in {time.time()-t0:.1f}s")

    if args.phase in ("a","all"): results["phase_a"] = run_phase_a(data)
    if args.phase in ("b","all"): results["phase_b"] = run_phase_b(data)
    if args.phase in ("c","all"): results["phase_c"] = run_phase_c(data)
    if args.phase in ("d","all"): results["phase_d"] = run_phase_d(data)
    if args.phase in ("e","all"): results["phase_e"] = run_phase_e()
    if args.phase in ("f","all"): results["phase_f"] = run_phase_f(data)

    primary = RESULTS_DIR / "EN8_2_pipeline_results.json"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = RESULTS_DIR / f"EN8_2_pipeline_results_{ts}.json"
    for p in [primary, archive]:
        with open(p, "w") as f:
            json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {primary}")


if __name__ == "__main__":
    main()
