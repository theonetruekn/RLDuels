import pytest
from pathlib import Path

from rlduels.src.create_video import create_videos_from_pair, generate_video_from_frames, _recreate_frames_from_trajectory, _recreate_frames_from_trajectory_with_seed, _recreate_frames_from_trajectory_no_seed
from rlduels.src.primitives.trajectory_pair import Transition, Trajectory, TrajectoryPair
from rlduels.src.utils.simulate import simulate_trajectories, simulate_trajectory_pairs
from rlduels.src.env_wrapper import EnvWrapper, GymWrapper

# Due to problems with the paths, this test is run manually without pytest
n = 1
name = "CartPole-v1"

env = GymWrapper.create_env(name=name, render_mode="rgb_array")
trajectory_pairs = simulate_trajectory_pairs(env, n)
for trajectory_pair in trajectory_pairs:
    result = create_videos_from_pair(trajectory_pair)
    
    assert result == "Success."
    assert isinstance(trajectory_pair.video1, Path)
    assert isinstance(trajectory_pair.video2, Path)
    assert trajectory_pair.video1.exists()
    assert trajectory_pair.video2.exists()
# Optional: Comment this out to look at the test-videos
trajectory_pair.video1.unlink()
trajectory_pair.video2.unlink()