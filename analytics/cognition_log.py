
import numpy as np


def build_cognition_trace(agent) -> dict:
    """Build a full cognition trace for a single agent.
    Returns a dict with all meta-cognitive, grammar, and ToM state."""

    # Level 1: Raw concepts
    concepts = agent.brain.get_concepts()
    concept_dict = {f"C{i}": float(concepts[i]) for i in range(min(16, len(concepts)))}

    # Level 2: Meta-concepts
    meta_info = agent.meta_concepts.introspect()

    # Level 3: Grammar
    grammar_roles = agent.grammar.get_role_descriptions()

    # Level 4: Theory of Mind
    tom_beliefs = agent.tom.describe_beliefs()

    # Level 5: Decision factors
    decision = {
        "action": agent._action.tolist() if hasattr(agent._action, 'tolist') else list(agent._action),
        "value": float(agent._value),
        "think_steps": agent._think_steps_used,
        "wm_accuracy": agent.brain.cumulative_wm_accuracy,
    }

    return {
        "agent_id": agent.id,
        "generation": agent.generation,
        "age": agent.ticks_alive,
        "total_reward": agent.total_reward,
        "concepts": concept_dict,
        "meta": {
            "pattern": meta_info["pattern"],
            "accuracy": meta_info["meta_wm_accuracy"],
            "active_dims": meta_info["active_dims"],
            "buffer_fill": meta_info["buffer_fill"],
        },
        "grammar": {
            "weight": agent.grammar.grammar_weight,
            "n_slots": agent.grammar.n_slots,
            "roles": grammar_roles,
        },
        "tom": tom_beliefs,
        "decision": decision,
    }


def format_cognition_report(agent) -> str:
    """Format a human-readable cognition report for an agent."""
    trace = build_cognition_trace(agent)
    lines = []

    lines.append(f"=== Agent #{trace['agent_id']} (gen={trace['generation']}, "
                 f"age={trace['age']}, reward={trace['total_reward']:.1f}) ===")

    # Concepts
    concepts = trace["concepts"]
    top_concepts = sorted(concepts.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    concept_str = ", ".join(f"{k}={v:.2f}" for k, v in top_concepts)
    lines.append(f"  CONCEPTS: {concept_str}")

    # Meta
    meta = trace["meta"]
    lines.append(f"  META: pattern={meta['pattern']}, accuracy={meta['accuracy']:.2f}, "
                 f"active_dims={meta['active_dims']}")

    # Grammar
    gram = trace["grammar"]
    if gram["weight"] > 0.01:
        role_strs = []
        for r in gram["roles"]:
            spec = r["specialization"]
            dims = r["top_dims"]
            role_strs.append(f"slot{r['slot']}(dims={dims}, spec={spec:.1f})")
        lines.append(f"  GRAMMAR (w={gram['weight']:.2f}): {', '.join(role_strs)}")

    # Theory of Mind
    tom = trace["tom"]
    if tom["tracked_count"] > 0:
        lines.append(f"  ToM ({tom['tracked_count']} tracked, acc={tom['mean_accuracy']:.2f}):")
        for aid, belief in list(tom["beliefs"].items())[:3]:
            pred = belief["predicted_concepts"]
            top_pred = sorted(enumerate(pred), key=lambda x: abs(x[1]), reverse=True)[:3]
            pred_str = ", ".join(f"C{i}={v:.2f}" for i, v in top_pred)
            lines.append(f"    Agent#{aid}: [{pred_str}] acc={belief['accuracy']:.2f}")

    # Decision
    dec = trace["decision"]
    action_names = ["move_x", "move_y", "mouth", "social", "manipulate", "signal"]
    action_strs = []
    for i, v in enumerate(dec["action"][:6]):
        if abs(v) > 0.3:
            name = action_names[i] if i < len(action_names) else f"a{i}"
            action_strs.append(f"{name}={v:.2f}")
    lines.append(f"  ACTION: {', '.join(action_strs) or 'idle'} "
                 f"(value={dec['value']:.2f}, think={dec['think_steps']})")

    return "\n".join(lines)
