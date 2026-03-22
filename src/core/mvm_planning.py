"""
Paper §III-E Multi-layer Valuemap (MVM) Planning: Φsem, Φdir, Φtrav -> Φtotal, gmid = argmax.
Wraps Graph.update_stage and get_nav_mode_and_goal_via_fuse (based on fuse_fields_and_extract_goal).
"""

from typing import Any, Tuple


def run_mvm_planning(graph: Any, loc_agent, vis: bool = True) -> Tuple[str, Any]:
    """
    Run MVM planning: assumes Graph is already updated via update_stage(map_info);
    selects gmid via fuse_fields_and_extract_goal.

    Args:
        graph: Graph instance (state updated by update_stage(map_info)).
        loc_agent: agent pose in sim coords (x, z, theta), used for direction field.
        vis: whether to save fused field visualization.

    Returns:
        (nav_mode, midterm_goal): navigation mode and mid-term goal (may be None).
    """
    return graph.get_nav_mode_and_goal_via_fuse(loc_agent, vis=vis)
