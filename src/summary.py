from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from .models import Conversation

def build_summary(conversations: List[Conversation]) -> Dict:
    intents = Counter([c.intent or "__NULL__" for c in conversations])
    nodes = Counter([c.node_id or "__NULL__" for c in conversations])
    escalated = Counter([str(c.is_escalated) for c in conversations])
    issue_types = Counter([c.issue_type or "__NULL__" for c in conversations])
    ratings = [c.rating for c in conversations if c.rating is not None]

    # stratification table (intent x node) top 20
    matrix = Counter([(c.intent or "__NULL__", c.node_id or "__NULL__") for c in conversations])
    top_matrix = [
        {"intent": k[0], "node_id": k[1], "count": v}
        for k, v in matrix.most_common(20)
    ]

    return {
        "n_conversations": len(conversations),
        "intents_top20": intents.most_common(20),
        "nodes_top20": nodes.most_common(20),
        "escalated_counts": escalated,
        "issue_types_top20": issue_types.most_common(20),
        "avg_rating": (sum(ratings) / len(ratings)) if ratings else None,
        "intent_node_top20": top_matrix,
    }
