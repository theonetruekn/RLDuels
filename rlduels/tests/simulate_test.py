import gymnasium as gym
from typing import List, Optional
from rlduels.src.primitives.trajectory_pair import Transition, Trajectory, TrajectoryPair, NDArray
from rlduels.src.env_wrapper import GymWrapper, EnvWrapper
from rlduels.src.utils.simulate import simulate_trajectories, simulate_trajectory_pairs

def test_simulate_trajectories():

    n = 5
    name = "CartPole-v1"
    
    env = GymWrapper.create_env(name=name, render_mode="rgb_array")
    trajectories = simulate_trajectories(env, n)

    assert len(trajectories) == n
    for trajectory in trajectories:
        assert trajectory.env_name == "CartPole-v1"
        assert trajectory.information['seed'] == 42
        assert len(trajectory.transitions) > 0
        for transition in trajectory.transitions:
            assert isinstance(transition, Transition)
            assert transition.state is not None
            assert transition.next_state is not None
            assert transition.reward is not None
            assert transition.action is not None

def test_simulate_trajectory_pairs():

    n = 5
    name = "CartPole-v1"
    
    env = GymWrapper.create_env(name=name, render_mode="rgb_array")
    trajectory_pairs = simulate_trajectory_pairs(env, n)

    assert len(trajectory_pairs) == n
    for pair in trajectory_pairs:
        assert isinstance(pair, TrajectoryPair)
        assert pair.trajectory1.env_name == name
        assert pair.trajectory2.env_name == name
        assert pair.trajectory1.information['seed'] == 42
        assert pair.trajectory2.information['seed'] == 42
        assert len(pair.trajectory1.transitions) > 0
        assert len(pair.trajectory2.transitions) > 0
        for transition in pair.trajectory1.transitions:
            assert isinstance(transition, Transition)
            assert transition.state is not None
            assert transition.next_state is not None
            assert transition.reward is not None
            assert transition.action is not None
        for transition in pair.trajectory2.transitions:
            assert isinstance(transition, Transition)
            assert transition.state is not None
            assert transition.next_state is not None
            assert transition.reward is not None
            assert transition.action is not None

