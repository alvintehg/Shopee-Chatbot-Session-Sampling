from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import Conversation

@dataclass
class SampleResult:
    conversations: List[Conversation]
    summary: Dict

def sample_random(conversations: List[Conversation], n: int, seed: int = 42) -> SampleResult:
    rng = random.Random(seed)
    n = min(n, len(conversations))
    picked = rng.sample(conversations, n) if n > 0 else []
    return SampleResult(picked, {"strategy": "random", "n": n, "population": len(conversations)})

def _key(c: Conversation, fields: Sequence[str]) -> Tuple[str, ...]:
    vals = []
    for f in fields:
        v = getattr(c, f, None)
        vals.append(str(v) if v is not None else "__NULL__")
    return tuple(vals)

def sample_stratified(
    conversations: List[Conversation],
    n: int,
    stratify_by: Sequence[str],
    max_per_stratum: Optional[int] = None,
    seed: int = 42,
) -> SampleResult:
    rng = random.Random(seed)
    buckets: Dict[Tuple[str, ...], List[Conversation]] = defaultdict(list)
    for c in conversations:
        buckets[_key(c, stratify_by)].append(c)

    # allocate evenly across strata then top-up
    strata = list(buckets.keys())
    rng.shuffle(strata)
    if not strata:
        return SampleResult([], {"strategy": "stratified", "population": len(conversations), "n": 0})

    base = max(1, n // len(strata))
    picked: List[Conversation] = []

    for k in strata:
        pool = buckets[k]
        take = min(base, len(pool))
        if max_per_stratum is not None:
            take = min(take, max_per_stratum)
        if take > 0:
            picked.extend(rng.sample(pool, take))

    # top-up to reach n
    remaining = n - len(picked)
    if remaining > 0:
        picked_ids = set([c.conversation_id for c in picked])
        remaining_pool = [c for c in conversations if c.conversation_id not in picked_ids]
        remaining = min(remaining, len(remaining_pool))
        if remaining > 0:
            picked.extend(rng.sample(remaining_pool, remaining))

    summary = {
        "strategy": "stratified",
        "population": len(conversations),
        "n": len(picked),
        "stratify_by": list(stratify_by),
        "n_strata": len(strata),
        "max_per_stratum": max_per_stratum,
    }
    return SampleResult(picked, summary)

def sample_top_k_then_random(
    conversations: List[Conversation],
    n: int,
    top_k_field: str,
    top_k: int,
    seed: int = 42,
) -> SampleResult:
    # Treat bool True as 1, False as 0, None as -1 (go last)
    def score(c: Conversation):
        v = getattr(c, top_k_field, None)
        if v is None:
            return -1
        if isinstance(v, bool):
            return 1 if v else 0
        try:
            return float(v)
        except Exception:
            return -1

    ranked = sorted(conversations, key=score, reverse=True)
    shortlist = ranked[: min(top_k, len(ranked))]
    return sample_random(shortlist, n, seed=seed)
