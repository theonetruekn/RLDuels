import numpy as np
import os

from collections import namedtuple
from typing import Any, List, NamedTuple, Tuple

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'terminated', 'truncated', 'next_state'])
Trajectory = namedtuple('Trajectory', ['initial_conditions', 'transitions'])

def compare_transitions(trans1, trans2) -> bool:
    """Compares two Transition named tuples to determine if they are equal.

    Args:
        trans1: The first Transition named tuple.
        trans2: The second Transition named tuple.

    Returns:
        bool: True if the transitions are equal, False otherwise.
    """
    return (np.array_equal(trans1.state, trans2.state) and
        np.array_equal(trans1.action, trans2.action) and
        np.array_equal(trans1.reward, trans2.reward) and
        trans1.terminated == trans2.terminated and
        trans1.truncated == trans2.truncated and
        np.allclose(trans1.next_state, trans2.next_state, equal_nan=True))

def compare_trajectories(trajectory1: Trajectory, trajectory2: Trajectory) -> bool:
    """Compares two Trajectory named tuples to determine if they are equal.

    Args:
        trajectory1: The first Trajectory named tuple.
        trajectory2: The second Trajectory named tuple.

    Returns:
        bool: True if the trajectories are equal, False otherwise.
    """
    # Compare initial conditions
    if not np.array_equal(trajectory1.initial_conditions, trajectory2.initial_conditions):
        return False

    # Compare the number of transitions and each transition
    if len(trajectory1.transitions) != len(trajectory2.transitions):
        return False

    for trans1, trans2 in zip(trajectory1.transitions, trajectory2.transitions):
        if not compare_transitions(trans1, trans2):
            return False

    return True

def get_reward(trajectory):
    return sum(reward for _, _, reward, _, _, _ in trajectory.transitions)

class TrajectoryPair:
    def __init__(self, trajectory1: Trajectory, trajectory2: Trajectory, preference=None):
        self._trajectory1 = trajectory1
        self._trajectory2 = trajectory2

        self._preference = preference
        self._skipped = False

        self._video1 = None
        self._video2 = None

    def prefer_video1(self):
        self._preference = 0
    
    def prefer_video2(self):
        self._preference = 1
    
    def prefer_no_video(self):
        self._preference = 0.5
    
    def skip(self):
        self._skipped = True
    
    def unskip(self):
        self._skipped = False

    def undo_preference(self):
        self._preference = None

    def set_video1(self, path):
        self._video1 = path
    
    def set_video2(self, path):
        self._video2 = path

    @property
    def trajectory1(self):
        return self._trajectory1

    @property
    def trajectory2(self):
        return self._trajectory2

    @property
    def preference(self):
        return self._preference

    @property
    def skipped(self):
        return self._skipped

    @property
    def video1(self):
        return self._video1
    
    @property
    def video2(self):
        return self._video2

    def delete_videos(self):
        try:
            os.remove(self._video1)
            os.remove(self._video2)
        except FileNotFoundError as e:
            print("Couldn't delete: ", e)


    def __eq__(self, other):
        if not isinstance(other, TrajectoryPair):
            return NotImplemented

        return (compare_trajectories(self.trajectory1, other.trajectory1) and 
                compare_trajectories(self.trajectory2, other.trajectory2))
    
    def to_bson(self):
        return {
            'trajectory1': serialize_trajectory(self._trajectory1),
            'trajectory2': serialize_trajectory(self._trajectory2),
            'preference': self._preference,
            'skipped': self._skipped
        }
    
    
def from_bson(bson_data) -> TrajectoryPair:
    """
    Factory method to create a TrajectoryPair instance from BSON data.

    Args:
        bson_data (dict): The BSON data (or a dictionary) representing a TrajectoryPair.

    Returns:
        TrajectoryPair: A new instance of TrajectoryPair initialized with data from the BSON.
    """
    trajectory1 = deserialize_trajectory(bson_data['trajectory1'])
    trajectory2 = deserialize_trajectory(bson_data['trajectory2'])
    preference = bson_data.get('preference', None)
    skipped = bson_data.get('skipped', False)

    pair = TrajectoryPair(trajectory1, trajectory2, preference)
    if skipped:
        pair.skip()
    else:
        pair.unskip()

    return pair
    
def convert_to_list(item):
    if isinstance(item, np.ndarray):
        return item.tolist(), str(item.dtype)
    elif isinstance(item, list):
        return [convert_to_list(subitem) for subitem in item], None
    elif isinstance(item, dict):
        return {key: convert_to_list(val)[0] for key, val in item.items()}, None
    return item, None

def serialize_transition(transition: Transition) -> dict:
    state_list, state_dtype = convert_to_list(transition.state)
    action_list, action_dtype = convert_to_list(transition.action)
    next_state_list, next_state_dtype = convert_to_list(transition.next_state)
    return {
        'state': state_list,
        'state_dtype': state_dtype,
        'action': action_list,
        'action_dtype': action_dtype,
        'reward': transition.reward,
        'terminated': transition.terminated,
        'truncated': transition.truncated,
        'next_state': next_state_list,
        'next_state_dtype': next_state_dtype
    }

def serialize_trajectory(trajectory: Trajectory) -> dict:
    transitions_dict = [serialize_transition(t) for t in trajectory.transitions]
    
    if not trajectory.initial_conditions:
        initial_conditions = {}
    else:
        seeds = [int(seed) for seed in trajectory.initial_conditions['seed']]
        
        # Update the initial_conditions with the converted seeds
        initial_conditions = trajectory.initial_conditions.copy()
        initial_conditions['seed'] = seeds

    return {
        'initial_conditions': initial_conditions,
        'transitions': transitions_dict
    }

def deserialize_transition(transition_dict: dict) -> Transition:
    return Transition(
        state=np.array(transition_dict['state'], dtype=transition_dict.get('state_dtype', 'float')),
        action=np.array(transition_dict['action'], dtype=transition_dict.get('action_dtype', 'float')),
        reward=transition_dict['reward'],
        terminated=transition_dict['terminated'],
        truncated=transition_dict['truncated'],
        next_state=np.array(transition_dict['next_state'], dtype=transition_dict.get('next_state_dtype', 'float'))
    )

def deserialize_trajectory(trajectory_dict: dict) -> Trajectory:
    transitions = [deserialize_transition(t) for t in trajectory_dict['transitions']]
    if 'initial_conditions' in trajectory_dict and trajectory_dict['initial_conditions'] is not None:
        initial_conditions = trajectory_dict['initial_conditions']
    else:
        initial_conditions = {}
    return Trajectory(
        initial_conditions=initial_conditions, 
        transitions=transitions
    )