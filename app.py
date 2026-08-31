"""HTTP surface for the demo.

Deliberately thin. The interesting behaviour lives in the library; this exposes three
views of it so a judge can watch the system work without a checkout:

  /world/{seed}   what the agent sees — opaque labels and nothing else
  /discover/{seed} a discovery run, including which hypotheses were refuted
  /evolve         one turn of the wheel: Gemini proposes, the gate decides
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, PlainTextResponse

sys.path.insert(0, str(Path(__file__).parent / "src"))

from metascience.config import load_env, model_name  # noqa: E402
from metascience.discovery import (run_discovery, score_on_held_out,  # noqa: E402
                                   simulation_data)
from metascience.auditor import audit_promotion  # noqa: E402
from metascience.evolution import (evaluate_detailed, evaluate_strategy,  # noqa: E402
                                   held_out_seeds, run_generation)
from metascience.experiment import (record_discovery, record_generation,  # noqa: E402
                                    summarise, to_csv_rows)
from metascience.gemini import TUNABLE, GeminiProposer, GeminiReasoner  # noqa: E402
from metascience.ledger import FileLedger, PromotionGate  # noqa: E402
from metascience.reasoner import HeuristicReasoner  # noqa: E402
from metascience.strategy import Strategy  # noqa: E402
from metascience.compose import generate_compound  # noqa: E402
from metascience.narrate import agent_brief, observer_narrative  # noqa: E402
from metascience.templates import generate  # noqa: E402

load_env()
app = FastAPI(title="meta-science", description="An agent that does science, and can be refuted.")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


def _ledger():
    """Firestore in the cloud, filesystem locally — the gate cannot tell the difference."""
    if os.environ.get("K_SERVICE"):          # set by Cloud Run
        from metascience.firestore_ledger import FirestoreLedger
        return FirestoreLedger(project=os.environ.get("GEMINI_PROJECT"))
    return FileLedger("runs")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """The interactive surface. Self-contained: no CDN, no external fetches."""
    page = Path(__file__).parent / "static" / "index.html"
    return page.read_text(encoding="utf-8")


@app.get("/evidence", response_class=HTMLResponse)
def evidence() -> str:
    """The frozen study, presented. Renders from static/study.json, not the live ledger."""
    return (Path(__file__).parent / "static" / "evidence.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    """Not /healthz: Google's frontend intercepts that path and returns its own 404."""
    return {"ok": True, "model": model_name()}


def _world(seed: int, depth: int, compound: bool, use_complex: bool = False):
    # `compound` is the legacy boolean from before depth existed; it means depth 1.
    # `use_complex` is a per-request toggle layered UNDER the deployment's master
    # switch: with METASCIENCE_COMPLEX unset the generator refuses, and that refusal
    # is surfaced honestly rather than masked.
    try:
        return generate_compound(seed, depth or (1 if compound else 0),
                                 include_complex=use_complex)
    except ValueError as exc:
        from fastapi import HTTPException
        raise HTTPException(422, str(exc)) from exc


@app.get("/world/{seed}")
def world(seed: int, depth: int = Query(0, ge=0, le=7), compound: bool = False,
          use_complex: bool = Query(False, alias="complex")) -> dict:
    """The agent's entire view. No structure, no domain terms, no ground truth."""
    w = _world(seed, depth, compound, use_complex)
    return w.describe() | {"brief": agent_brief(w)}


@app.get("/world/{seed}/truth")
def world_truth(seed: int, depth: int = Query(0, ge=0, le=7), compound: bool = False,
                use_complex: bool = Query(False, alias="complex")) -> dict:
    """OBSERVER VIEW — ground truth for humans studying the benchmark.

    Public on purpose: the repo is public and any reader can compute this from the
    seed, so secrecy is not what protects the benchmark. The boundary is process-level
    — nothing on the agent's path calls this — and anonymisation guards against
    retrieval from training, not against a person reading the answer.
    """
    w = _world(seed, depth, compound, use_complex)
    return {
        "world_id": w.world_id,
        "seed": w.seed,
        "observable": list(w.observable),
        "hidden": list(w.hidden),
        "ground_truth": w.ground_truth(),
        "narrative": observer_narrative(w),
        "exogenous": {n: {"mean": node.exo_mean, "sd": node.exo_sd}
                      for n, node in w.nodes.items() if node.mechanism is None},
        "noise": {n: node.mechanism.noise
                  for n, node in w.nodes.items() if node.mechanism},
    }


@app.get("/world/{seed}/inspect", response_class=HTMLResponse)
def world_inspect(seed: int) -> str:
    return (Path(__file__).parent / "static" / "world.html").read_text(encoding="utf-8")


@app.get("/discover/{seed}")
def discover(seed: int, live: bool = False, depth: int = Query(0, ge=0, le=7),
             use_complex: bool = Query(False, alias="complex"),
             include_data: bool = False) -> dict:
    w = _world(seed, depth, False, use_complex)
    reasoner = GeminiReasoner() if live else HeuristicReasoner()
    run = run_discovery(w, reasoner, Strategy(), seed=seed)
    variables = list(w.observable)
    probes = [(variables[0], 0.5), (variables[0], -0.5)]
    held_out = score_on_held_out(w, run, probes, seed=seed + 5000)
    rec = record_discovery(_ledger(), w, run, held_out, Strategy(),
                           models={"reasoner": model_name() if live else "heuristic"},
                           notes=(f"depth={depth} complex={use_complex}"
                                  if (depth or use_complex) else ""))
    out = {
        **run.to_dict(),
        "held_out_score": held_out,
        "reasoner": "gemini" if live else "heuristic",
        "run_id": rec.run_id,
    }
    if include_data:
        # Re-derived from the same seeds the run used — identical by determinism,
        # not by having been remembered. See simulation_data.
        out["simulation"] = simulation_data(w, run, Strategy(), seed=seed)
    return out


def _canon_strategy(ledger) -> tuple[Strategy, list[tuple[dict, float]]]:
    """The strategy currently in canon, and the diffs the gate has refused.

    Promotions move canon; refusals do not. Replaying the promoted diffs in
    order therefore reconstructs the live champion, and the refused ones become
    the proposer's memory.
    """
    params: dict = {}
    refused: list[tuple[dict, float]] = []
    try:
        records = sorted(ledger.receipts(), key=lambda r: r.get("created_at", 0))
    except Exception:                      # a cold ledger is not an error
        return Strategy(), refused
    for r in records:
        diff = r.get("diff") or {}
        if not diff:
            continue
        if r.get("verdict") == "PROMOTED":
            params.update({k: v[0] for k, v in diff.items() if k in TUNABLE})
        else:
            refused.append((diff, r.get("challenger_score", 0.0)
                            - r.get("champion_score", 0.0)))
    try:
        return Strategy(**params), refused
    except TypeError:                      # an unknown knob must not break the demo
        return Strategy(), refused


@app.post("/evolve")
def evolve() -> dict:
    """Gemini proposes a change to the scientist's own method; the gate rules on it."""
    seeds = held_out_seeds(24)
    ledger = _ledger()
    gate = PromotionGate(
        ledger, evaluate_strategy, margin=0.02, detailed=evaluate_detailed,
        auditor=lambda r: audit_promotion(r, {k: t.__name__ for k, t in TUNABLE.items()}))
    # Continue from canon rather than restarting at champion-v1 every click.
    # set_canon only stores a name and digest, so the live champion is rebuilt by
    # replaying the promoted diffs in order — the same reconstruction
    # tests/test_committed_receipts.py uses to prove the receipts replay.
    champion, refused = _canon_strategy(ledger)
    challenger_holder = {}
    proposer = GeminiProposer()
    # Verdicts already on the record are handed back, so the proposer does not
    # re-offer something the gate has already turned down.
    for diff, gain in refused[-5:]:
        proposer.remember_verdict(diff, gain, promoted=False)
    original = proposer.propose

    def capture(champ, notes=""):
        challenger_holder["c"] = original(champ, notes)
        return challenger_holder["c"]

    proposer.propose = capture
    _, receipt = run_generation(
        ledger, gate, champion, proposer, seeds,
        "accuracy is saturated on most worlds; 12 experiments x 400 samples each")
    rec = record_generation(ledger, receipt, champion, challenger_holder["c"], seeds,
                            models={"proposer": model_name(),
                                    "auditor": "gemini-3.5-flash-lite"})
    return {
        "verdict": receipt.verdict,
        "candidate": receipt.candidate,
        "diff": receipt.diff,
        "champion_score": receipt.champion_score,
        "challenger_score": receipt.challenger_score,
        "margin_required": receipt.margin_required,
        "reason": receipt.reason,
        "digest": receipt.digest(),
        "held_out_worlds": len(seeds),
        "score_parts": {"champion": receipt.champion_parts,
                        "challenger": receipt.challenger_parts},
        "audit": receipt.audit,
        "run_id": rec.run_id,
    }


@app.get("/receipts")
def receipts() -> dict:
    led = _ledger()
    rows = led.receipts() if hasattr(led, "receipts") else []
    return {"count": len(rows), "receipts": rows[-20:]}


# -- the evidence base -------------------------------------------------------

@app.get("/experiments")
def experiments(limit: int = Query(100, le=500)) -> dict:
    """Every recorded run, newest last. Each one carries what it needs to be rerun."""
    rows = _ledger().experiments(limit=limit)
    return {"count": len(rows), "experiments": rows}


@app.get("/stats")
def stats() -> dict:
    """Population-level figures over everything recorded — the table a paper opens with.

    Computed from stored records rather than by rerunning the code, so it describes what
    was actually observed and not what the current version would produce today.
    """
    return summarise(_ledger().experiments(limit=500))


@app.get("/export.csv", response_class=PlainTextResponse)
def export_csv() -> str:
    """One row per hypothesis tested, joined to ground truth — the long format analysis
    actually wants. Straight into pandas or R without reshaping."""
    import csv
    import io

    rows = to_csv_rows(_ledger().experiments(limit=500))
    if not rows:
        return "no experiments recorded yet\n"
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()
