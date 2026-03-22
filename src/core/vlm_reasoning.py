"""
Paper §III-D VLM-based High-level Reasoning: GSSL + goal -> semantic scores αᵢ, nav mode (Explore/Find/Judge).
Wraps Graph.GSSL_gen and analyze_navigation_status.
"""

from typing import Any, Dict, Optional


def run_vlm_reasoning(
    graph: Any,
    loc_agent,
    current_img=None,
    goal_img=None,
) -> Optional[Dict]:
    """
    Run VLM reasoning: generate GSSL, score unexplored entities, set NavMode and direction.

    Args:
        graph: Graph instance (must have GSSL_gen, analyze_navigation_status).
        loc_agent: agent pose in sim coords (x, z, theta).
        current_img: current observation image (optional, for VLM multimodal).
        goal_img: goal image (optional).

    Returns:
        plan_result or None if VLM not invoked.
    """
    graph.GSSL_gen(loc_agent)
    if current_img is not None and goal_img is not None:
        return graph.analyze_navigation_status(current_img, goal_img)
    return None
