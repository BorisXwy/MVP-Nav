"""
Paper §III-C Physical Perception: monocular sequence -> GSSL (3D OBB, pseudo-depth, VGGT + Grounded-SAM).
Wraps Map.run_stage as a single entry point.
"""

from typing import Any, Dict, Type


def run_physical_perception(
    MapClass: Type,
    set2: int,
    set1: int,
    current_step: int,
    stage: int,
    model_info: Dict[str, Any],
    args: Any,
    agent: Any,
    loc_agent_sim,
) -> tuple:
    """
    Run physical perception for this stage: mapping, point cloud, OBB, BEV and fields -> map_info.
    Builds obs provider from agent so Map does not depend on Agent type.

    Returns:
        (map_inst, map_info): Map instance and map_info dict for VLM/MVM.
    """
    obs_provider = _obs_provider_from_agent(agent)
    map_inst = MapClass(set2, set1, current_step, stage, model_info, args, obs_provider)
    map_info = map_inst.run_stage(loc_agent_sim)
    return map_inst, map_info


def _obs_provider_from_agent(agent: Any) -> Any:
    """Build observation data interface from agent for Map (Map does not depend on Agent type)."""
    from types import SimpleNamespace
    return SimpleNamespace(
        obs_history=getattr(agent, "obs_history", []),
        node_segment_result=getattr(agent, "node_segment_result", []),
        main_segment_result=getattr(agent, "main_segment_result", []),
        sub_segment_result=getattr(agent, "sub_segment_result", []),
        segment_save_folder=getattr(agent, "segment_save_folder", ""),
    )
