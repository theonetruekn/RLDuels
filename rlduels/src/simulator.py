#TODO: Refactor

import math
from typing import Any, List, Tuple

import gymnasium as gym
import numpy as np
from src.DataHandling.trajectory_pair import Transition, Trajectory
from src.DataHandling.agent import Agent

''' A class for simulating agents on gym environments and returning trajectories.
'''
class Simulator():
    
    def __init__(self, env, agent=None, frame_rate: int = 50, run_speed_factor: float = 1.0):
        self.env = env
        self.agent = agent
        self.frame_rate = frame_rate
        self.run_speed_factor = run_speed_factor

    def simulate_for_n_seconds(self, n: int = 10, seed=42) -> Tuple[List[np.ndarray], Trajectory]:
        """
        Simulates the environment for a specified number of seconds and records the state transitions and frames.

        This method runs the simulation for a given number of seconds, taking actions in the environment either 
        randomly or using a provided agent. Each step in the environment is recorded as a frame and a transition. 
        The simulation can be reproducible by setting a seed value. If the simulation terminates or gets truncated, 
        it is reset with a new seed. The seeds are captured such that a trajectory can be reconstructed.

        Parameters:
        - n (int, optional): The duration, in seconds, for which to run the simulation. Defaults to 10 seconds.
        - seed (int, optional): The seed for the random number generator, ensuring reproducibility. Defaults to 42.

        Returns:
        - Tuple[List[np.ndarray], Trajectory]: A tuple containing two elements:
            1. A list of numpy arrays representing the frames captured during the simulation.
            2. A Trajectory object encapsulating the initial conditions (the seeds) and the sequence of transitions 
            (state, action, reward, termination status, and next state) observed in the simulation.
        """
        frames = []
        transitions = []
        rng = np.random.default_rng(seed)
        observation, info = self.env.reset(seed = int(seed))
        initial_condition = {'seed': [seed]}

        # flooring necessary, as run_speed_factor is float 
        max_frames = math.floor(n * self.frame_rate * self.run_speed_factor)

        for _ in range(max_frames):
            frames.append(self.env.render())

            if self.agent is None:
                action = self.env.action_space.sample()
            else:
                action = self.agent.generate_action()
                        
            current_obs = observation
            observation, reward, terminated, truncated, info = self.env.step(action)

            transitions.append(Transition(current_obs, action, reward, terminated, truncated, observation))

            if terminated or truncated:
                seed = rng.integers(0, 100)
                self.env.reset(seed = int(seed))
                initial_condition['seed'].append(seed)

        return frames, Trajectory(initial_condition, transitions)

    def new_simulate_for_n_seconds(self, seconds: int = 10, seed=42) -> Tuple[List[np.ndarray], Trajectory]:
        frames = []
        transitions = []
        rng = np.random.default_rng(seed)
        observation, _ = self.env.reset(seed = int(seed))
        initial_condition = {'seed': None}

        # flooring necessary, as run_speed_factor is float 
        max_frames = math.floor(seconds * self.frame_rate * self.run_speed_factor)

        for _ in range(max_frames):
            frames.append(self.env.render())

            if self.agent is None:
                action = self.env.action_space.sample()
            else:
                action = self.agent.generate_action()
                        
            current_obs = observation
            observation, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                seed = rng.integers(0, 100)
                observation, _ = self.env.reset(seed = seed)
            
            transitions.append(Transition(current_obs, action, reward, terminated, truncated, observation))


        return frames, Trajectory(initial_condition, transitions)

    
    def simulate_episode(self, agent:Agent = None, seed: int = 42, max_time_steps: int = 1000) -> Tuple[List[np.ndarray], Trajectory]:
        frames = []
        transitions = []
        observation, info = self.env.reset(seed=seed)
        initial_condition = {'seed': [seed]}

        for _ in range(max_time_steps):
            frames.append(self.env.render())
            if agent is None:
                action = self.env.action_space.sample()
            else:
                action = agent.generate_action()
            
            current_obs = observation

            observation, reward, terminated, truncated, _ = self.env.step(action)

            transitions.append(Transition(current_obs, action, reward, terminated, truncated, observation))

            if terminated or truncated:
                break
        
        trajectory = Trajectory(initial_conditions, transitions)

        return frames, trajectory

