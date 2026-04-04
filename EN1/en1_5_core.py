"""EN1.5 Core — Deduction Under Uncertainty.

Pure computation module (no I/O, no API calls).

Three-dataset evaluation of SL deduction vs scalar probability
multiplication, measuring calibration (ECE, Brier) and uncertainty
preservation across evidence levels N ∈ {5, 10, 50, 100, 1000}.

Datasets:
  - ASIA (Lauritzen & Spiegelhalter 1988): 8-node canonical BN
  - ALARM (Beinlich et al. 1989): 37-node standard BN benchmark
  - Synthea (Walonoski et al. 2018): empirical conditionals from
    ~1K synthetic FHIR R4 patient bundles (Condition→Condition
    comorbidity edges)

Key hypothesis:
  SL deduction preserves calibration under sparse evidence because
  uncertainty in the antecedent opinion pulls the deduced projected
  probability toward the base rate (Bayesian shrinkage). Scalar
  multiplication of point estimates has no such regularization and
  becomes overconfident at low N.

Mathematical foundation:
  SL deduction (Jøsang 2016 §12.6):
    c_y = b_x·c_{y|x} + d_x·c_{y|¬x} + u_x·(a_x·c_{y|x} + ā_x·c_{y|¬x})
  for each component c ∈ {b, d, u}.

  Scalar law of total probability:
    P̂(Y) = p̂(X)·P(Y|X) + (1−p̂(X))·P(Y|¬X)

  At low N, p̂(X) has high variance → scalar P̂(Y) inherits that
  variance with no uncertainty signal. SL deduction propagates
  antecedent uncertainty into u_y, pulling P(ω_y) toward a_y.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Path setup for sibling imports ──
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_EXPERIMENTS_ROOT = _SCRIPT_DIR.parent
if str(_EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_ROOT))

_REPO_ROOT = _EXPERIMENTS_ROOT.parent  # jsonld-ex root

from jsonld_ex.confidence_algebra import Opinion, deduce


# ═══════════════════════════════════════════════════════════════════
# 1. Data classes
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DeductionEdge:
    """A single parent→child edge with ground-truth conditionals.

    Represents the binary deduction:
      P(child=positive | parent=positive) and
      P(child=positive | parent=negative)
    along with the marginal P(parent=positive).

    For BN models, "positive" = state 0 (binarized).
    For Synthea, "positive" = condition present.
    """
    parent: str
    child: str
    p_parent: float              # P(parent = positive)
    p_child_given_parent: float  # P(child = positive | parent = positive)
    p_child_given_not_parent: float  # P(child = positive | parent = negative)

    @property
    def p_child(self) -> float:
        """Ground truth P(child = positive) via law of total probability."""
        return (
            self.p_parent * self.p_child_given_parent
            + (1 - self.p_parent) * self.p_child_given_not_parent
        )


@dataclass
class DeductionKB:
    """A knowledge base of deduction edges from a single source."""
    name: str
    nodes: List[str]
    edges: List[DeductionEdge]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════
# 2. KB Loaders — BN models (ASIA, ALARM)
# ═══════════════════════════════════════════════════════════════════

# Evidence level for conditional opinions (high → near-dogmatic).
# This isolates the effect of antecedent uncertainty.
_CONDITIONAL_EVIDENCE = 10_000

# All pgmpy built-in BN models, ordered by size.
ALL_BN_MODELS = [
    "cancer",       #   5 nodes,   4 edges
    "earthquake",   #   5 nodes,   4 edges
    "survey",       #   6 nodes,   6 edges
    "asia",         #   8 nodes,   8 edges
    "sachs",        #  11 nodes,  17 edges
    "child",        #  20 nodes,  25 edges
    "insurance",    #  27 nodes,  52 edges
    "water",        #  32 nodes,  66 edges
    "mildew",       #  35 nodes,  46 edges
    "alarm",        #  37 nodes,  46 edges
    "barley",       #  48 nodes,  84 edges
    "hailfinder",   #  56 nodes,  66 edges
    "hepar2",       #  70 nodes, 123 edges
    "win95pts",     #  76 nodes, 112 edges
    "andes",        # 223 nodes, 338 edges
]


def load_bn_kb(model_name: str) -> DeductionKB:
    """Load a pgmpy built-in BN and extract binary deduction edges.

    For each directed edge (parent, child) in the DAG:
    1. Compute P(parent=state0) via Variable Elimination.
    2. Compute P(child=state0 | parent=state0) via VE with evidence.
    3. Derive P(child=state0 | parent≠state0) from total probability.

    Multi-valued nodes are binarized: first state vs. all other states.
    pgmpy uses string state names (e.g., 'yes'/'no', 'TRUE'/'FALSE',
    'LOW'/'NORMAL'/'HIGH'), so we discover them from the CPDs.

    Multi-parent nodes are handled correctly because VE marginalizes
    over all non-evidence variables.

    Edges where P(parent=state0) < 0.001 or > 0.999 are skipped
    (degenerate — no meaningful deduction possible).
    """
    from pgmpy.utils import get_example_model
    from pgmpy.inference import VariableElimination

    model = get_example_model(model_name)
    infer = VariableElimination(model)

    nodes = sorted(model.nodes())

    # Discover state names for each node from CPDs
    state_names: Dict[str, list] = {}
    for cpd in model.get_cpds():
        var = cpd.variable
        # pgmpy stores state names in cpd.state_names dict
        state_names[var] = cpd.state_names[var]

    # Cache marginals to avoid redundant VE queries
    marginals: Dict[str, np.ndarray] = {}
    for node in nodes:
        result = infer.query([node])
        marginals[node] = result.values

    edges: List[DeductionEdge] = []
    for parent, child in model.edges():
        # Use the first state as "positive"
        parent_state0 = state_names[parent][0]
        child_state0 = state_names[child][0]

        p_parent_0 = float(marginals[parent][0])

        # Skip degenerate edges
        if p_parent_0 < 0.001 or p_parent_0 > 0.999:
            continue

        # P(child=state0 | parent=state0)
        result_given = infer.query(
            [child], evidence={parent: parent_state0}
        )
        p_child_given_0 = float(result_given.values[0])

        # P(child=state0) — marginal
        p_child_0 = float(marginals[child][0])

        # Derive P(child=state0 | parent≠state0) from total probability:
        #   P(C=0) = P(P=0)·P(C=0|P=0) + P(P≠0)·P(C=0|P≠0)
        p_parent_not0 = 1.0 - p_parent_0
        if p_parent_not0 < 1e-12:
            continue
        p_child_given_not0 = (
            (p_child_0 - p_parent_0 * p_child_given_0) / p_parent_not0
        )
        # Clamp to [0, 1] for numerical safety
        p_child_given_not0 = float(np.clip(p_child_given_not0, 0.0, 1.0))

        edges.append(DeductionEdge(
            parent=parent,
            child=child,
            p_parent=p_parent_0,
            p_child_given_parent=p_child_given_0,
            p_child_given_not_parent=p_child_given_not0,
        ))

    return DeductionKB(
        name=model_name,
        nodes=nodes,
        edges=edges,
        metadata={
            "source": f"pgmpy.utils.get_example_model('{model_name}')",
            "n_nodes_original": len(nodes),
            "n_edges_original": len(model.edges()),
            "n_edges_after_filter": len(edges),
            "binarization": "first_state_vs_rest",
            "state_names_sample": {
                k: v for k, v in list(state_names.items())[:5]
            },
        },
    )


def load_asia_kb() -> DeductionKB:
    """Load the ASIA BN (Lauritzen & Spiegelhalter 1988)."""
    return load_bn_kb("asia")


def load_alarm_kb() -> DeductionKB:
    """Load the ALARM BN (Beinlich et al. 1989)."""
    return load_bn_kb("alarm")


def load_all_bn_kbs(
    model_names: Optional[List[str]] = None,
) -> List[DeductionKB]:
    """Load multiple pgmpy BN models, skipping any that fail.

    Args:
        model_names: List of model names. Default: ALL_BN_MODELS.

    Returns:
        List of successfully loaded DeductionKB objects.
    """
    names = model_names or ALL_BN_MODELS
    kbs: List[DeductionKB] = []
    for name in names:
        try:
            kb = load_bn_kb(name)
            kbs.append(kb)
        except Exception as e:
            # Some models may fail VE (treewidth too high, etc.)
            import warnings
            warnings.warn(f"Skipping model '{name}': {e}")
    return kbs


# ═══════════════════════════════════════════════════════════════════
# 3. KB Loader — Synthea (FHIR R4 bundles)
# ═══════════════════════════════════════════════════════════════════

_SYNTHEA_DIR = _REPO_ROOT / "data" / "synthea" / "fhir_r4"

# Minimum thresholds for Synthea edge extraction
_MIN_CONDITION_PATIENTS = 20    # condition must appear in ≥20 patients
_MIN_COOCCURRENCE = 10          # both conditions must co-occur in ≥10 patients
_MAX_CONDITION_PREVALENCE = 0.95  # skip near-universal conditions


def load_synthea_kb() -> DeductionKB:
    """Load empirical conditionals from Synthea FHIR R4 bundles.

    Extracts Condition→Condition (comorbidity) edges. For each patient:
    1. Collect the set of unique Condition SNOMED codes.
    2. Build a patient × condition binary matrix.
    3. For each ordered pair (A, B) with sufficient support:
       - P(B | A) = n(A ∧ B) / n(A)
       - P(B | ¬A) = n(¬A ∧ B) / n(¬A)
       - P(A) = n(A) / N

    Only pairs with sufficient co-occurrence and non-degenerate
    conditionals are included.
    """
    if not _SYNTHEA_DIR.exists():
        raise FileNotFoundError(
            f"Synthea data not found at {_SYNTHEA_DIR}. "
            f"Run: python tools/download_synthea.py"
        )

    json_files = sorted(_SYNTHEA_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in {_SYNTHEA_DIR}")

    # ── Step 1: Extract per-patient condition sets ──
    patient_conditions: List[set] = []
    all_conditions: set = set()

    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                bundle = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        if bundle.get("resourceType") != "Bundle":
            continue

        conditions = set()
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") != "Condition":
                continue
            code_obj = resource.get("code", {})
            codings = code_obj.get("coding", [])
            for coding in codings:
                if coding.get("system") == "http://snomed.info/sct":
                    display = coding.get("display", "")
                    code = coding.get("code", "")
                    if display and code:
                        # Use "code:display" as key for uniqueness + readability
                        key = f"{code}:{display}"
                        conditions.add(key)
                        all_conditions.add(key)
                    break  # one SNOMED code per condition

        patient_conditions.append(conditions)

    n_patients = len(patient_conditions)
    if n_patients < 50:
        raise ValueError(
            f"Only {n_patients} patients found — need ≥50 for meaningful "
            f"conditional estimates."
        )

    # ── Step 2: Filter to sufficiently prevalent conditions ──
    from collections import Counter
    cond_counts = Counter()
    for conds in patient_conditions:
        for c in conds:
            cond_counts[c] += 1

    # Keep conditions with prevalence in [_MIN_CONDITION_PATIENTS, max_prev * N]
    max_count = int(_MAX_CONDITION_PREVALENCE * n_patients)
    valid_conditions = sorted([
        c for c, count in cond_counts.items()
        if _MIN_CONDITION_PATIENTS <= count <= max_count
    ])

    if len(valid_conditions) < 5:
        raise ValueError(
            f"Only {len(valid_conditions)} conditions meet prevalence "
            f"threshold — need ≥5."
        )

    # ── Step 3: Build co-occurrence matrix and extract edges ──
    # For efficiency, encode as indices
    cond_to_idx = {c: i for i, c in enumerate(valid_conditions)}
    n_conds = len(valid_conditions)

    # Binary patient × condition matrix
    matrix = np.zeros((n_patients, n_conds), dtype=bool)
    for i, conds in enumerate(patient_conditions):
        for c in conds:
            if c in cond_to_idx:
                matrix[i, cond_to_idx[c]] = True

    # Prevalence vector
    prevalence = matrix.sum(axis=0)  # shape (n_conds,)

    edges: List[DeductionEdge] = []
    nodes_used: set = set()

    for a_idx in range(n_conds):
        n_a = int(prevalence[a_idx])
        n_not_a = n_patients - n_a
        if n_a < _MIN_CONDITION_PATIENTS or n_not_a < _MIN_CONDITION_PATIENTS:
            continue

        p_a = n_a / n_patients
        mask_a = matrix[:, a_idx]
        mask_not_a = ~mask_a

        for b_idx in range(n_conds):
            if a_idx == b_idx:
                continue

            # Co-occurrence count
            n_ab = int((mask_a & matrix[:, b_idx]).sum())
            if n_ab < _MIN_COOCCURRENCE:
                continue

            n_not_a_b = int((mask_not_a & matrix[:, b_idx]).sum())

            p_b_given_a = n_ab / n_a
            p_b_given_not_a = n_not_a_b / n_not_a

            # Skip degenerate conditionals
            if p_b_given_a < 0.001 or p_b_given_a > 0.999:
                continue
            if p_b_given_not_a < 0.001 or p_b_given_not_a > 0.999:
                continue

            # Only keep edges where conditionals differ meaningfully
            # (otherwise deduction is trivial)
            if abs(p_b_given_a - p_b_given_not_a) < 0.01:
                continue

            cond_a = valid_conditions[a_idx]
            cond_b = valid_conditions[b_idx]

            edges.append(DeductionEdge(
                parent=cond_a,
                child=cond_b,
                p_parent=p_a,
                p_child_given_parent=p_b_given_a,
                p_child_given_not_parent=p_b_given_not_a,
            ))
            nodes_used.add(cond_a)
            nodes_used.add(cond_b)

    if len(edges) < 10:
        raise ValueError(
            f"Only {len(edges)} edges extracted from Synthea — need ≥10. "
            f"Try lowering thresholds or adding more patient data."
        )

    return DeductionKB(
        name="synthea",
        nodes=sorted(nodes_used),
        edges=edges,
        metadata={
            "source": str(_SYNTHEA_DIR),
            "n_patients": n_patients,
            "n_conditions_total": len(all_conditions),
            "n_conditions_valid": len(valid_conditions),
            "n_edges": len(edges),
            "min_condition_patients": _MIN_CONDITION_PATIENTS,
            "min_cooccurrence": _MIN_COOCCURRENCE,
            "edge_type": "Condition→Condition (comorbidity)",
        },
    )


# ═══════════════════════════════════════════════════════════════════
# 4. Single deduction trial
# ═══════════════════════════════════════════════════════════════════

def run_deduction_trial(
    edge: DeductionEdge,
    n_evidence: int,
    seed: int,
) -> Dict[str, Any]:
    """Run a single deduction trial comparing SL vs scalar.

    Protocol:
    1. Draw N observations of the parent from Bernoulli(P_true(parent)).
    2. Scalar: p̂ = r/N, then P̂(child) via total probability.
    3. SL: Opinion.from_evidence(r, N-r), then deduce().
    4. Draw a binary outcome from Bernoulli(P_true(child)) for calibration.

    Conditional opinions are built with high evidence (_CONDITIONAL_EVIDENCE)
    to isolate the effect of antecedent uncertainty.

    Args:
        edge: The deduction edge with ground-truth conditionals.
        n_evidence: Number of observations N for the parent.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: scalar_pred, sl_pred, sl_uncertainty,
        ground_truth_p, outcome.
    """
    rng = np.random.RandomState(seed)

    # ── Step 1: Simulate parent observations ──
    observations = rng.binomial(1, edge.p_parent, size=n_evidence)
    r = int(observations.sum())       # positive evidence (parent = positive)
    s = n_evidence - r                # negative evidence

    # ── Step 2: Scalar baseline ──
    # Point estimate of P(parent = positive)
    p_hat = r / n_evidence
    scalar_pred = (
        p_hat * edge.p_child_given_parent
        + (1 - p_hat) * edge.p_child_given_not_parent
    )

    # ── Step 3: SL deduction ──
    # Antecedent opinion: uncertain, based on limited evidence
    opinion_x = Opinion.from_evidence(
        positive=r,
        negative=s,
        prior_weight=2.0,
        base_rate=edge.p_parent,
    )

    # Conditional opinions: high evidence (near-dogmatic)
    # P(child|parent): treat p as fraction of positive evidence
    cond_r = edge.p_child_given_parent * _CONDITIONAL_EVIDENCE
    cond_s = _CONDITIONAL_EVIDENCE - cond_r
    opinion_y_given_x = Opinion.from_evidence(
        positive=cond_r,
        negative=cond_s,
        prior_weight=2.0,
        base_rate=edge.p_child,  # use marginal as base rate
    )

    cond_nr = edge.p_child_given_not_parent * _CONDITIONAL_EVIDENCE
    cond_ns = _CONDITIONAL_EVIDENCE - cond_nr
    opinion_y_given_not_x = Opinion.from_evidence(
        positive=cond_nr,
        negative=cond_ns,
        prior_weight=2.0,
        base_rate=edge.p_child,
    )

    # Deduce
    deduced = deduce(opinion_x, opinion_y_given_x, opinion_y_given_not_x)
    sl_pred = deduced.projected_probability()
    sl_uncertainty = deduced.uncertainty

    # ── Step 4: Ground truth and binary outcome ──
    ground_truth_p = edge.p_child  # exact via total probability
    outcome = int(rng.binomial(1, ground_truth_p))

    return {
        "scalar_pred": float(scalar_pred),
        "sl_pred": float(sl_pred),
        "sl_uncertainty": float(sl_uncertainty),
        "sl_antecedent_uncertainty": float(opinion_x.uncertainty),
        "ground_truth_p": float(ground_truth_p),
        "outcome": outcome,
    }


# ═══════════════════════════════════════════════════════════════════
# 5. Calibration metrics
# ═══════════════════════════════════════════════════════════════════

def compute_calibration_metrics(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
    *,
    ground_truth_probs: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute ECE, Brier score, and binned calibration data.

    Args:
        predictions: Predicted probabilities, shape (N,).
        outcomes: Binary outcomes (0 or 1), shape (N,).
        n_bins: Number of bins for ECE computation.
        ground_truth_probs: If provided, also compute MAE vs true P.

    Returns:
        Dict with keys: ece, brier, bin_edges, bin_counts, bin_accs,
        bin_confs. If ground_truth_probs given, also mae_vs_true.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    n = len(predictions)

    # ── Brier score: mean squared error ──
    brier = float(np.mean((predictions - outcomes) ** 2))

    # ── ECE: Expected Calibration Error ──
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_counts = np.zeros(n_bins, dtype=int)
    bin_accs = np.full(n_bins, np.nan)
    bin_confs = np.full(n_bins, np.nan)

    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (predictions >= lo) & (predictions < hi)
        else:
            # Last bin includes right edge
            mask = (predictions >= lo) & (predictions <= hi)

        count = int(mask.sum())
        bin_counts[i] = count
        if count > 0:
            acc = float(outcomes[mask].mean())
            conf = float(predictions[mask].mean())
            bin_accs[i] = acc
            bin_confs[i] = conf
            ece += (count / n) * abs(acc - conf)

    result: Dict[str, Any] = {
        "ece": float(ece),
        "brier": brier,
        "bin_edges": bin_edges.tolist(),
        "bin_counts": bin_counts.tolist(),
        "bin_accs": bin_accs.tolist(),
        "bin_confs": bin_confs.tolist(),
    }

    if ground_truth_probs is not None:
        gt = np.asarray(ground_truth_probs, dtype=np.float64)
        result["mae_vs_true"] = float(np.mean(np.abs(predictions - gt)))

    return result


# ═══════════════════════════════════════════════════════════════════
# 6. Evidence sweep — full protocol
# ═══════════════════════════════════════════════════════════════════

def run_evidence_sweep(
    kb: DeductionKB,
    n_values: List[int],
    n_reps: int = 1000,
    seed: int = 42,
) -> Dict[int, Dict[str, Any]]:
    """Run the full deduction experiment across evidence levels.

    For each N in n_values:
      For each edge in kb.edges:
        For each rep in range(n_reps):
          Run a deduction trial → collect scalar_pred, sl_pred, outcome.
      Aggregate into calibration metrics.

    Args:
        kb: Knowledge base with edges.
        n_values: Evidence levels to test.
        n_reps: Repetitions per (edge, N) combination.
        seed: Base seed (actual seed = seed + edge_idx * 100000 + n * 1000 + rep).

    Returns:
        Dict mapping N → aggregated metrics dict.
    """
    results: Dict[int, Dict[str, Any]] = {}

    for n_ev in n_values:
        all_scalar_preds = []
        all_sl_preds = []
        all_sl_uncertainties = []
        all_sl_antecedent_uncertainties = []
        all_outcomes = []
        all_ground_truths = []

        for edge_idx, edge in enumerate(kb.edges):
            for rep in range(n_reps):
                trial_seed = seed + edge_idx * 100_000 + n_ev * 1000 + rep
                trial = run_deduction_trial(edge, n_evidence=n_ev, seed=trial_seed)
                all_scalar_preds.append(trial["scalar_pred"])
                all_sl_preds.append(trial["sl_pred"])
                all_sl_uncertainties.append(trial["sl_uncertainty"])
                all_sl_antecedent_uncertainties.append(trial["sl_antecedent_uncertainty"])
                all_outcomes.append(trial["outcome"])
                all_ground_truths.append(trial["ground_truth_p"])

        scalar_preds = np.array(all_scalar_preds)
        sl_preds = np.array(all_sl_preds)
        outcomes = np.array(all_outcomes)
        ground_truths = np.array(all_ground_truths)

        scalar_metrics = compute_calibration_metrics(
            scalar_preds, outcomes, n_bins=10,
            ground_truth_probs=ground_truths,
        )
        sl_metrics = compute_calibration_metrics(
            sl_preds, outcomes, n_bins=10,
            ground_truth_probs=ground_truths,
        )

        results[n_ev] = {
            "scalar_ece": scalar_metrics["ece"],
            "sl_ece": sl_metrics["ece"],
            "scalar_brier": scalar_metrics["brier"],
            "sl_brier": sl_metrics["brier"],
            "scalar_mae": scalar_metrics["mae_vs_true"],
            "sl_mae": sl_metrics["mae_vs_true"],
            "sl_mean_uncertainty": float(np.mean(all_sl_uncertainties)),
            "sl_mean_antecedent_uncertainty": float(np.mean(all_sl_antecedent_uncertainties)),
            "n_trials": len(all_scalar_preds),
            # Keep detailed metrics for plotting
            "scalar_calibration": scalar_metrics,
            "sl_calibration": sl_metrics,
        }

    return results


# ═════════════════════════════════════════════════════════════════
# 7. Multi-hop path extraction and chained deduction
# ═════════════════════════════════════════════════════════════════


def extract_paths(
    kb: DeductionKB,
    max_length: int = 4,
) -> List[List[DeductionEdge]]:
    """Extract all directed acyclic paths of length 2..max_length.

    A path of length L consists of L consecutive edges:
      A→B→C is length 2 (2 edges, 3 nodes).

    Paths are acyclic: no node appears more than once.

    Args:
        kb: Knowledge base with edges.
        max_length: Maximum number of edges in a path (default 4).

    Returns:
        List of paths, where each path is a list of DeductionEdge.
    """
    from collections import defaultdict

    # Build adjacency: parent -> list of edges
    adj: Dict[str, List[DeductionEdge]] = defaultdict(list)
    for edge in kb.edges:
        adj[edge.parent].append(edge)

    all_paths: List[List[DeductionEdge]] = []

    def _extend(current_path: List[DeductionEdge], visited: set) -> None:
        """DFS to enumerate all acyclic paths."""
        if len(current_path) >= 2:
            all_paths.append(list(current_path))
        if len(current_path) >= max_length:
            return
        last_child = current_path[-1].child
        for next_edge in adj.get(last_child, []):
            if next_edge.child not in visited:
                current_path.append(next_edge)
                visited.add(next_edge.child)
                _extend(current_path, visited)
                current_path.pop()
                visited.discard(next_edge.child)

    # Start a path from each edge
    for start_edge in kb.edges:
        visited = {start_edge.parent, start_edge.child}
        _extend([start_edge], visited)

    return all_paths


def run_multihop_deduction_trial(
    path: List[DeductionEdge],
    n_evidence: int,
    seed: int,
    conditional_evidence: int = _CONDITIONAL_EVIDENCE,
) -> Dict[str, Any]:
    """Run a single chained deduction trial along a multi-hop path.

    Protocol:
    1. Draw N observations of the first parent from Bernoulli(P_true).
    2. Chain through each edge:
       - Scalar: apply total probability with point estimate
       - SL: apply deduce() using the previous deduced opinion
    3. Compare final predictions against ground truth P(last child).

    Args:
        path: Ordered list of consecutive DeductionEdges.
        n_evidence: Observations N for the first antecedent.
        seed: Random seed.
        conditional_evidence: Evidence level for conditional opinions.
            Use 10000 (default) to isolate antecedent uncertainty.
            Use lower values (e.g., 100) to simulate uncertain KB.

    Returns:
        Dict with scalar_pred, sl_pred, sl_uncertainty,
        sl_antecedent_uncertainty, ground_truth_p, outcome,
        path_length, path_description.
    """
    rng = np.random.RandomState(seed)
    first_edge = path[0]

    # ── Step 1: Simulate first antecedent observations ──
    observations = rng.binomial(1, first_edge.p_parent, size=n_evidence)
    r = int(observations.sum())
    s = n_evidence - r

    # Scalar: point estimate of P(parent)
    p_hat = r / n_evidence

    # SL: opinion from evidence
    opinion_current = Opinion.from_evidence(
        positive=r,
        negative=s,
        prior_weight=2.0,
        base_rate=first_edge.p_parent,
    )
    initial_antecedent_u = opinion_current.uncertainty

    # ── Step 2: Chain through each edge ──
    for edge in path:
        # Build conditional opinions
        cond_r = edge.p_child_given_parent * conditional_evidence
        cond_s = conditional_evidence - cond_r
        opinion_y_given_x = Opinion.from_evidence(
            positive=cond_r,
            negative=cond_s,
            prior_weight=2.0,
            base_rate=edge.p_child,
        )

        cond_nr = edge.p_child_given_not_parent * conditional_evidence
        cond_ns = conditional_evidence - cond_nr
        opinion_y_given_not_x = Opinion.from_evidence(
            positive=cond_nr,
            negative=cond_ns,
            prior_weight=2.0,
            base_rate=edge.p_child,
        )

        # SL deduction
        opinion_current = deduce(
            opinion_current, opinion_y_given_x, opinion_y_given_not_x
        )

        # Scalar chain
        p_hat = (
            p_hat * edge.p_child_given_parent
            + (1 - p_hat) * edge.p_child_given_not_parent
        )

    # ── Step 3: Ground truth and outcome ──
    last_edge = path[-1]
    ground_truth_p = last_edge.p_child  # BN marginal
    outcome = int(rng.binomial(1, ground_truth_p))

    # Path description for logging
    nodes = [path[0].parent] + [e.child for e in path]
    path_desc = " → ".join(nodes)

    return {
        "scalar_pred": float(p_hat),
        "sl_pred": float(opinion_current.projected_probability()),
        "sl_uncertainty": float(opinion_current.uncertainty),
        "sl_antecedent_uncertainty": float(initial_antecedent_u),
        "ground_truth_p": float(ground_truth_p),
        "outcome": outcome,
        "path_length": len(path),
        "path_description": path_desc,
    }


def run_multihop_sweep(
    kb: DeductionKB,
    n_values: List[int],
    n_reps: int = 1000,
    seed: int = 42,
    max_path_length: int = 4,
    max_paths_per_length: int = 200,
    conditional_evidence: int = _CONDITIONAL_EVIDENCE,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Run multi-hop deduction experiment grouped by path length.

    For each path length L in [2..max_path_length]:
      For each path of length L (up to max_paths_per_length):
        For each N in n_values:
          For each rep:
            Run chained deduction trial.
      Aggregate into calibration metrics.

    Args:
        kb: Knowledge base with edges.
        n_values: Evidence levels.
        n_reps: Repetitions per (path, N).
        seed: Base seed.
        max_path_length: Max edges in a path.
        max_paths_per_length: Cap paths per length to control runtime.
        conditional_evidence: Evidence for conditional opinions.

    Returns:
        Dict mapping "length_L" -> {N -> metrics_dict}.
    """
    all_paths = extract_paths(kb, max_length=max_path_length)

    # Group by length
    from collections import defaultdict
    paths_by_length: Dict[int, List[List[DeductionEdge]]] = defaultdict(list)
    for path in all_paths:
        paths_by_length[len(path)].append(path)

    results: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for length in sorted(paths_by_length.keys()):
        paths = paths_by_length[length]
        # Cap if needed
        if len(paths) > max_paths_per_length:
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(paths), max_paths_per_length, replace=False)
            paths = [paths[i] for i in sorted(indices)]

        length_results: Dict[int, Dict[str, Any]] = {}

        for n_ev in n_values:
            all_scalar = []
            all_sl = []
            all_outcomes = []
            all_gt = []
            all_sl_u = []

            for path_idx, path in enumerate(paths):
                for rep in range(n_reps):
                    trial_seed = (
                        seed + length * 1_000_000
                        + path_idx * 10_000
                        + n_ev * 100 + rep
                    )
                    trial = run_multihop_deduction_trial(
                        path, n_evidence=n_ev, seed=trial_seed,
                        conditional_evidence=conditional_evidence,
                    )
                    all_scalar.append(trial["scalar_pred"])
                    all_sl.append(trial["sl_pred"])
                    all_outcomes.append(trial["outcome"])
                    all_gt.append(trial["ground_truth_p"])
                    all_sl_u.append(trial["sl_uncertainty"])

            scalar_preds = np.array(all_scalar)
            sl_preds = np.array(all_sl)
            outcomes = np.array(all_outcomes)
            gt = np.array(all_gt)

            scalar_metrics = compute_calibration_metrics(
                scalar_preds, outcomes, n_bins=10, ground_truth_probs=gt,
            )
            sl_metrics = compute_calibration_metrics(
                sl_preds, outcomes, n_bins=10, ground_truth_probs=gt,
            )

            length_results[n_ev] = {
                "scalar_ece": scalar_metrics["ece"],
                "sl_ece": sl_metrics["ece"],
                "scalar_brier": scalar_metrics["brier"],
                "sl_brier": sl_metrics["brier"],
                "scalar_mae": scalar_metrics["mae_vs_true"],
                "sl_mae": sl_metrics["mae_vs_true"],
                "sl_mean_uncertainty": float(np.mean(all_sl_u)),
                "n_paths": len(paths),
                "n_trials": len(all_scalar),
            }

        results[f"length_{length}"] = length_results

    return results
