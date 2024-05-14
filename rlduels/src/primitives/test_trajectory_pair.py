import numpy as np
import pytest
from pydantic import ValidationError
from rlduels.src.primitives.trajectory_pair import NDArray, TrajectoryPair, Transition, Trajectory
from rlduels.src.utils.simulate import simulate_trajectories, simulate_trajectory_pairs

def test_ndarray_with_trajectory_simulation():
    
    environment_name = "CartPole-v1"  # Example environment
    trajectories = simulate_trajectories(env=environment_name, n=1) # Simulate the data
    trajectory = trajectories[0]

    
    for transition in trajectory.transitions:
        
        for attribute in ['state', 'action', 'next_state']:
            ndarray_instance = getattr(transition, attribute)

            assert isinstance(ndarray_instance, NDArray), "Should be an instance of NDArray"
            
            assert isinstance(ndarray_instance.array, np.ndarray), "The .array attribute should be a numpy array"



def test_transition_integrity():
    environment_name = "CartPole-v1"  # Example environment
    trajectories = simulate_trajectories(n=1)  # Simulate the data
    for transition in trajectories[0].transitions:
        assert isinstance(transition, Transition), "Each transition should be an instance of Transition"
        assert isinstance(transition.state, NDArray) and isinstance(transition.next_state, NDArray), "States should be NDArrays"
        assert transition.reward is not None, "Reward should be captured"


def test_trajectory_consistency():
    environment_name = "CartPole-v1"  # Example environment
    trajectories = simulate_trajectories(env=environment_name, n=1)  # Simulate the data
    trajectory = trajectories[0]
    assert trajectory.env_name == environment_name, "Environment name should match the simulation input"
    assert len(trajectory.transitions) > 0, "There should be multiple transitions"
    assert all(isinstance(t, Transition) for t in trajectory.transitions), "All elements should be transitions"



def test_trajectory_pairing():
    environment_name = "CartPole-v1"  # Example environment
    trajectory_pairs = simulate_trajectory_pairs(env=environment_name, n=2) # Simulate the data
    for pair in trajectory_pairs:
        assert pair.trajectory1.env_name == environment_name and pair.trajectory2.env_name == environment_name, "Both trajectories in a pair should be from the specified environment"
        assert isinstance(pair.trajectory1, Trajectory) and isinstance(pair.trajectory2, Trajectory), "Both should be Trajectory instances"

