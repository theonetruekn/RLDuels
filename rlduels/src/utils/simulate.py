import gymnasium as gym
import numpy as np
from typing import List, Optional, Tuple
from rlduels.src.primitives.trajectory_pair import Transition, Trajectory, TrajectoryPair, NDArray
from rlduels.src.env_wrapper import EnvWrapper, GymWrapper

def simulate_one_episode(env: EnvWrapper, seed: int) -> Trajectory:
    """
    Simulates a single episode in the environment and records the state transitions.

    Parameters:
    - env (EnvWrapper): The environment to simulate.
    - seed (int): The seed for the random number generator, ensuring reproducibility.

    Returns:
    - Trajectory: A Trajectory object encapsulating the sequence of transitions 
      (state, action, reward, termination status, and next state) observed in the episode.
    """
    observation, info = env.reset(seed=seed)
    transitions: List[Transition] = []
    done = False

    while not done:
        env.render()
        action = env.sample_action()
        next_observation, reward, terminated, truncated, info = env.step(action)

        transitions.append(Transition.create(
            state=observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            next_state=next_observation
        ))

        observation = next_observation
        done = terminated or truncated

    return Trajectory(
        env_name=env.name,
        information={'seed': seed},
        transitions=transitions
    )

def simulate_trajectories(env: EnvWrapper = GymWrapper.create_env(name="CartPole-v1", render_mode="rgb_array"), 
                          n: int = 5, seed: Optional[int] = 42) -> List[Trajectory]:
    """
    Simulates multiple episodes in the environment and records the state transitions.

    Parameters:
    - env (EnvWrapper): The environment to simulate.
    - n (int): The number of episodes to simulate. Defaults to 5.
    - seed (Optional[int]): The seed for the random number generator, ensuring reproducibility. Defaults to 42.

    Returns:
    - List[Trajectory]: A list of Trajectory objects, each encapsulating the sequence of transitions 
      (state, action, reward, termination status, and next state) observed in an episode.
    """
    rng = np.random.default_rng(seed)
    trajectories: List[Trajectory] = []

    for episode in range(n):
        episode_seed = int(rng.integers(0, 10000))
        trajectory = simulate_one_episode(env, episode_seed)
        trajectories.append(trajectory)

    env.close()
    return trajectories

def simulate_trajectory_pairs(env: EnvWrapper = GymWrapper.create_env(name="CartPole-v1", render_mode="rgb_array"), n: int = 5, seed: Optional[int] = 42) -> List[TrajectoryPair]:
    trajectories: List[Trajectory] = simulate_trajectories(env, 2 * n, seed)
    trajectory_pairs: List[TrajectoryPair] = []

    for i in range(0, len(trajectories), 2):
        trajectory_pair = TrajectoryPair(
            trajectory1=trajectories[i],
            trajectory2=trajectories[i + 1]
        )
        trajectory_pairs.append(trajectory_pair)

    return trajectory_pairs

def simulate_for_n_seconds(env: EnvWrapper = GymWrapper.create_env(name="CartPole-v1", render_mode="rgb_array"), 
                           seconds: int = 10, seed: Optional[int] = 42) -> Tuple[List[NDArray], Trajectory]:
    """
    Simulates the environment for a specified number of seconds and records the state transitions and frames.

    This method runs the simulation for a given number of seconds, taking actions in the environment either 
    randomly or using a provided agent. Each step in the environment is recorded as a frame and a transition. 
    The simulation can be reproducible by setting a seed value. If the simulation terminates or gets truncated, 
    it is reset with a new seed. The seeds are captured such that a trajectory can be reconstructed.

    Parameters:
    - seconds (int, optional): The duration, in seconds, for which to run the simulation. Defaults to 10 seconds.
    - seed (int, optional): The seed for the random number generator, ensuring reproducibility. Defaults to 42.

    Returns:
    - Tuple[List[NDArray], Trajectory]: A tuple containing two elements:
        1. A list of numpy arrays representing the frames captured during the simulation.
        2. A Trajectory object encapsulating the initial conditions (the seeds) and the sequence of transitions 
        (state, action, reward, termination status, and next state) observed in the simulation.
    """
    rng = np.random.default_rng(seed)
    frames: List[NDArray] = []
    transitions: List[Transition] = []

    observation, info = env.reset(seed=seed)
    initial_condition = {'seed': seed}
    
    max_frames = math.floor(seconds * env.frame_rate * env.run_speed_factor)
    done = False
    step_count = 0

    while step_count < max_frames:
        frames.append(env.render())
        action = env.sample_action()
        next_observation, reward, terminated, truncated, info = env.step(action)

        transitions.append(Transition.create(
            state=observation,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            next_state=next_observation
        ))

        observation = next_observation
        done = terminated or truncated
        step_count += 1

        if done:
            new_seed = rng.integers(0, 10000)
            observation, info = env.reset(seed=new_seed)
            done = False

    trajectory = Trajectory(
        env_name=env.name,
        information=initial_condition,
        transitions=transitions
    )

    env.close()
    return frames, trajectory