"""
Layer 3 (policy half): Model-Fit Engine

Given the current pooled capacity, determine which curated/bundled models
are currently runnable, and how much more pooled capacity is needed to
unlock the next tier. This directly powers the companion app's
"unlock more models as you connect more devices" visualization.

IMPORTANT: capacity figures here are *rough sizing heuristics*, not a
substitute for real benchmarking. A model "fitting" numerically does not
guarantee acceptable tokens/sec -- see docs/MODEL_NOTES.md.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelSpec:
    id: str
    display_name: str
    tier: str                  # "starter" | "pro" | "max"
    params_b: float            # parameters in billions
    quant: str                 # e.g. "4-bit"
    min_pooled_mb: int         # rough total pooled memory needed
    min_nodes: int             # minimum device count required
    use_case: str              # "chat" | "code"
    notes: str = ""


def load_model_catalog(path: str) -> List[ModelSpec]:
    with open(path, "r") as f:
        raw = json.load(f)
    return [ModelSpec(**m) for m in raw]


def fit_report(pooled_mb: int, node_count: int, catalog: List[ModelSpec]) -> dict:
    """Return which models currently fit, which are locked, and the gap
    to the next unlockable model (by memory and by node count)."""
    runnable, locked = [], []
    for m in sorted(catalog, key=lambda x: x.min_pooled_mb):
        fits_mem = pooled_mb >= m.min_pooled_mb
        fits_nodes = node_count >= m.min_nodes
        entry = {
            "id": m.id,
            "display_name": m.display_name,
            "tier": m.tier,
            "use_case": m.use_case,
            "min_pooled_mb": m.min_pooled_mb,
            "min_nodes": m.min_nodes,
        }
        if fits_mem and fits_nodes:
            runnable.append(entry)
        else:
            gap_mb = max(0, m.min_pooled_mb - pooled_mb)
            gap_nodes = max(0, m.min_nodes - node_count)
            entry["gap_mb"] = gap_mb
            entry["gap_nodes"] = gap_nodes
            locked.append(entry)

    next_unlock: Optional[dict] = None
    if locked:
        next_unlock = min(locked, key=lambda x: x["gap_mb"])

    return {
        "pooled_mb": pooled_mb,
        "node_count": node_count,
        "runnable": runnable,
        "locked": locked,
        "next_unlock": next_unlock,
    }
