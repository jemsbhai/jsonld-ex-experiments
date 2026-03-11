def _make_associativity_test(
    br: float,
    deviations: list[float],
    bdu_deviations: list[float],
    use_non_dogmatic: bool = True,
) -> Callable:
    """Factory: create a @given-decorated associativity test for a fixed base rate.

    Hypothesis rejects @given on functions with default arguments, so we
    use a factory that closes over the base rate value instead.

    Args:
        br: Fixed base rate for all three opinions.
        deviations: Shared list to accumulate max deviations.
        bdu_deviations: Shared list to accumulate BDU deviations.
        use_non_dogmatic: If True, use non_dogmatic_opinion_strategy to
            avoid the dogmatic fallback branch (which is known to break
            associativity).  Josang's associativity proof (S12.3) requires
            at least one non-dogmatic opinion in each pair.
    """
    strat = non_dogmatic_opinion_strategy if use_non_dogmatic else opinion_strategy

    @given(
        a=strat(base_rate=br),
        b=strat(base_rate=br),
        c=strat(base_rate=br),
    )
    @PROP_SETTINGS
    def _assoc(a: Opinion, b: Opinion, c: Opinion) -> None:
        left = cumulative_fuse(cumulative_fuse(a, b), c)
        right = cumulative_fuse(a, cumulative_fuse(b, c))
        dev = opinion_max_deviation(left, right)
        bdu_dev = opinion_bdu_max_deviation(left, right)
        deviations.append(dev)
        bdu_deviations.append(bdu_dev)
        assert dev < FLOAT_TOL, (
            f"Associativity violated at base_rate={br}: max_dev={dev}, "
            f"a={a}, b={b}, c={c}, left={left}, right={right}"
        )

    return _assoc


def test_p3_fusion_associativity_equal_base_rates() -> dict[str, Any]:
    """Verify cumulative fusion associativity when all base rates are equal.

    Josang (2016, S12.3) proves associativity for the standard kappa-based
    formula, which requires at least one non-dogmatic opinion (u > 0) in
    each fused pair.  When both opinions are dogmatic (u = 0), the formula
    has a 0/0 indeterminate form and falls back to simple averaging with
    equal relative dogmatism (gamma_A = gamma_B = 0.5).  Simple averaging
    is NOT associative: avg(avg(a,b),c) != avg(a,avg(b,c)) in general.

    We verify:
      (a) Non-dogmatic case: associativity holds (PASS expected).
      (b) Dogmatic case: associativity does NOT hold (documented as finding).

    We test across 7 fixed base rates for robustness.
    """
    deviations: list[float] = []
    bdu_deviations: list[float] = []
    dogmatic_deviations: list[float] = []
    dogmatic_counterexamples: list[dict[str, Any]] = []

    fixed_base_rates = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    sub_results: dict[str, str] = {}

    # ── (a) Non-dogmatic: associativity should hold ──
    for br in fixed_base_rates:
        test_fn = _make_associativity_test(
            br, deviations, bdu_deviations, use_non_dogmatic=True,
        )
        try:
            test_fn()
            sub_results[f"non_dogmatic_br={br}"] = "PASS"
        except Exception as e:
            sub_results[f"non_dogmatic_br={br}"] = f"FAIL: {e}"

    # ── (b) Dogmatic: document non-associativity ──
    # Use explicit canonical examples rather than random generation to
    # clearly demonstrate the mathematical phenomenon.
    dogmatic_test_cases = [
        # (a, b, c) triples — all dogmatic, same base_rate
        (Opinion(0.7, 0.3, 0.0, 0.5), Opinion(0.6, 0.4, 0.0, 0.5), Opinion(1.0, 0.0, 0.0, 0.5)),
        (Opinion(0.3, 0.7, 0.0, 0.5), Opinion(0.8, 0.2, 0.0, 0.5), Opinion(0.5, 0.5, 0.0, 0.5)),
        (Opinion(0.9, 0.1, 0.0, 0.5), Opinion(0.1, 0.9, 0.0, 0.5), Opinion(0.9, 0.1, 0.0, 0.5)),
        (Opinion(1.0, 0.0, 0.0, 0.5), Opinion(0.0, 1.0, 0.0, 0.5), Opinion(0.5, 0.5, 0.0, 0.5)),
    ]
    n_dogmatic_violations = 0
    for a_op, b_op, c_op in dogmatic_test_cases:
        left = cumulative_fuse(cumulative_fuse(a_op, b_op), c_op)
        right = cumulative_fuse(a_op, cumulative_fuse(b_op, c_op))
        dev = opinion_max_deviation(left, right)
        dogmatic_deviations.append(dev)
        if dev > FLOAT_TOL:
            n_dogmatic_violations += 1
            if len(dogmatic_counterexamples) < 5:
                dogmatic_counterexamples.append({
                    "a": {"b": a_op.belief, "d": a_op.disbelief, "u": a_op.uncertainty},
                    "b": {"b": b_op.belief, "d": b_op.disbelief, "u": b_op.uncertainty},
                    "c": {"b": c_op.belief, "d": c_op.disbelief, "u": c_op.uncertainty},
                    "left": {"b": left.belief, "d": left.disbelief, "u": left.uncertainty},
                    "right": {"b": right.belief, "d": right.disbelief, "u": right.uncertainty},
                    "max_deviation": dev,
                })

    sub_results["dogmatic_non_associativity_documented"] = (
        f"CONFIRMED: {n_dogmatic_violations}/{len(dogmatic_test_cases)} "
        f"dogmatic triples violate associativity"
    )

    import numpy as np
    non_dogmatic_passed = all(
        v == "PASS"
        for k, v in sub_results.items()
        if k.startswith("non_dogmatic_br=")
    )

    return {
        "outcome": "PASS" if non_dogmatic_passed else "FAIL",
        "n_tested": len(deviations) + len(dogmatic_test_cases),
        "max_deviation": float(np.max(deviations)) if deviations else 0.0,
        "mean_deviation": float(np.mean(deviations)) if deviations else 0.0,
        "std_deviation": float(np.std(deviations)) if deviations else 0.0,
        "counterexamples": dogmatic_counterexamples if dogmatic_counterexamples else None,
        "sub_properties": sub_results,
        "notes": (
            f"Tested {len(fixed_base_rates)} fixed base rates with non-dogmatic opinions, "
            f"{len(deviations)} total non-dogmatic examples. "
            f"BDU max_dev={float(np.max(bdu_deviations)) if bdu_deviations else 0:.2e}. "
            f"FINDING: Associativity holds for non-dogmatic opinions (standard kappa formula) "
            f"but fails for dogmatic opinions (equal-weight averaging fallback). "
            f"Dogmatic violations: {n_dogmatic_violations}/{len(dogmatic_test_cases)}, "
            f"max_dev={max(dogmatic_deviations) if dogmatic_deviations else 0:.4f}. "
            f"This is mathematically inherent: simple averaging is not associative."
        ),
    }
