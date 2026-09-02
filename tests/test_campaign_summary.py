"""Pin the n=30 campaign's published numbers to the committed data.

Repo convention (see test_published_figures.py): any number the papers quote
must fail the build if the committed artifact drifts. summary.json is derived
by scripts/campaign_analysis.py from runs/campaign/ (committed on this branch).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / "docs/campaign/summary.json").read_text())


def test_grid_complete():
    for arm in ["a-evolution", "b1-recall", "b2-refine", "b3-thirdlaw", "b4-secondlaw"]:
        reps = list((ROOT / "runs/campaign" / arm).glob("rep-*.json"))
        assert len(reps) == 30, f"{arm}: {len(reps)} reps"


def test_b1_recall_rates():
    b1 = S["b1_recall"]
    assert (b1["tycho_blind"]["n"], b1["tycho_blind"]["recognises"]) == (30, 2)
    assert (b1["tycho_labeled"]["n"], b1["tycho_labeled"]["recognises"]) == (30, 30)
    assert (b1["synth_blind"]["n"], b1["synth_blind"]["recognises"]) == (30, 11)
    # Wilson 95% bounds as published (3 sig figs)
    assert round(b1["tycho_blind"]["rate"][2], 3) == 0.213
    assert round(b1["tycho_labeled"]["rate"][1], 3) == 0.886


def test_b2_refinement_precondition_never_fired():
    b2 = S["b2_refine"]
    assert b2["round1_shapes"]["cosine"] == 30
    assert b2["round2_ran"] == 0  # all round-1 proposals extrapolated already


def test_b3_thirdlaw():
    b3 = S["b3_thirdlaw"]
    assert (b3["power_law_form"], b3["named_a_3_2_law"]) == (29, 13)
    assert round(b3["exponent_mean"], 4) == 1.7056
    assert b3["exponent_min"] >= 1.5 and b3["exponent_max"] <= 2.0


def test_b4_keeps():
    b4 = S["b4_secondlaw"]
    assert (b4["equal_areas"]["kept_in_round2"], b4["equal_areas"]["final_extrapolates"]) == (2, 22)
    assert (b4["drag"]["kept_in_round2"], b4["drag"]["final_extrapolates"]) == (11, 11)


def test_a_evolution_pattern():
    a = S["a_evolution"]
    assert a["verdict_patterns"] == {"PRR": 29, "PRP": 1}
    assert (a["refused_higher_scoring"], a["refused_lower_scoring"]) == (33, 26)
    assert a["audit_on_promotions"] == {"flagged": 27, "legitimate": 4, "other": 0}
