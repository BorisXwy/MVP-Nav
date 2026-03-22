"""
MVP-Nav single-episode navigation loop (paper Algorithm 1).
Delegates to src.core.navigation.run_episode, aligned with core four layers.
"""

from src.core.navigation import run_episode as _run_episode

__all__ = ["run_episode"]


def run_episode(
    episode_idx,
    envs,
    agent,
    graph,
    MapClass,
    model_info,
    args,
    *,
    use_vlm_per_stage=False,
    obs_save_folder=None,
    depth_save_folder=None,
    segment_save_folder=None,
    midterm_save_folder=None,
    shorterm_save_folder=None,
    result_file_handle=None,
):
    """
    Run a single episode (see core.navigation.run_episode).
    """
    return _run_episode(
        episode_idx,
        envs,
        agent,
        graph,
        MapClass,
        model_info,
        args,
        use_vlm_per_stage=use_vlm_per_stage,
        obs_save_folder=obs_save_folder,
        depth_save_folder=depth_save_folder,
        segment_save_folder=segment_save_folder,
        midterm_save_folder=midterm_save_folder,
        shorterm_save_folder=shorterm_save_folder,
        result_file_handle=result_file_handle,
    )
