import datetime
import math
import time

import cv2
import gymnasium as gym
import numpy as np
import yaml

from pathlib import Path
from typing import Any, List, Tuple

from rlduels.src.primitives.trajectory_pair import Transition, Trajectory, TrajectoryPair
from rlduels.src.env_wrapper import EnvWrapper, GymWrapper

CONFIG_PATH = 'config.yaml'
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)

video_folder = config["VIDEO_FOLDER"]
frame_rate = config["FRAME_RATE"]
run_speed_factor = config["RUN_SPEED_FACTOR"]

env_pool = {}


def create_video_from_pair(trajectory_pair: TrajectoryPair):
    """
    Generates videos for a pair of trajectories.

    Parameters:
    trajectory_pair: TrajectoryPair, a TrajectoryPair object.

    Returns:
    tuple: A tuple containing file paths for the two generated videos.
    """
    env_name = trajectory_pair.env_name
    if env_name not in env_pool.keys():
        print("Creating env!")
        env_pool[env_name] =  GymWrapper.create_env(name = env_name, render_mode="rgb_array")
        print(env_pool)
        print("Env created")
    
    env: EnvWrapper = env_pool[env_name]

    t1: Trajectory = trajectory_pair.trajectory1
    t2: Trajectory = trajectory_pair.trajectory2

    frames1 = _recreate_frames_from_trajectory(t1, env)
    print("Frames1:", frames1)
    frames2 = _recreate_frames_from_trajectory(t2, env)

    file1: Path = generate_video_from_frames(frames1)
    file2: Path = generate_video_from_frames(frames2)

    return file1, file2

def generate_video_from_frames(frames: List[np.ndarray], file_name: str = "trajectory", add_timestamp: bool = True) -> str:
    '''
    Generates a video from a list of frames.

    Parameters:
    frames: List[np.ndarray], a list of frames (as numpy arrays) to be included in the video.
    file_name: str, the base name for the video file. Defaults to "trajectory".
    add_timestamp: bool, whether to add a timestamp to the file name. Defaults to True.

    Returns:
    str: The path to the generated video file.
    '''
    if not frames:
        raise ValueError("The frames list is empty. Cannot generate video.")

    height, width, layers = frames[0].shape

    if layers != 3:
        raise ValueError("Frames are not an RGB-array.")

    file_name = f"{video_folder}/{file_name}"

    if add_timestamp:
        file_name = f"{file_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.webm"
    
    fps = frame_rate * run_speed_factor

    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    video = cv2.VideoWriter(file_name, fourcc, fps, (width, height))

    for frame in frames:
        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video.release()

    return Path(file_name)

def _recreate_frames_from_trajectory(trajectory: Trajectory, env: EnvWrapper) -> List[np.ndarray]:
    '''
    Recreates frames from a given trajectory.

    Parameters:
    trajectory: Trajectory, a trajectory object.

    Returns:
    List[np.ndarray]: A list of frames recreated from the trajectory.
    '''
    if not trajectory.information.get('seed'):
        return _recreate_frames_from_trajectory_no_seed(trajectory, env)
    else:
        return _recreate_frames_from_trajectory_with_seed(trajectory, env)

def _recreate_frames_from_trajectory_with_seed(trajectory: Trajectory, env: EnvWrapper) -> List[np.ndarray]:
    information = trajectory.information
    transitions = trajectory.transitions
    seed = information['seed']
    env.reset(seed=seed)
    frames = []
    ctr = 0
    for t in transitions:
        state, action, _, terminated, truncated, _ = t.unpack()
        print("action: ", action)
        env.step(action) 
        frames.append(env.render())

        if terminated or truncated:
            env.reset(seed=seed)

    return frames

def _recreate_frames_from_trajectory_no_seed(trajectory: Trajectory, env: EnvWrapper) -> List[np.ndarray]:
    
    _, transitions = trajectory
    frames = []
    starting_state = transitions[0].state
    obs, info = env.reset(obs=starting_state)
    
    for _, action, _, terminated, truncated, next_obs in transitions:
        if terminated or truncated:
            env.reset(obs=next_obs)
        else:
            env.step(action) 
        frames.append(env.render())

    return frames
    
    