import datetime
import math
from typing import Any, List, Tuple

import cv2
import time
import gymnasium as gym
import numpy as np
import yaml
from src.DataHandling.trajectory_pair import Transition, Trajectory, TrajectoryPair

class VideoExtractor():
    '''
    VideoExtractor is a class for generating videos from trajectories in Gym environments. 
    It assumes the Gym environment is set to 'rgb_array' render mode and facilitates the 
    creation of video files from the trajectories of environment states.
    '''

    def __init__(self, env, video_folder: str, frame_rate: int = 50, run_speed_factor: float = 1.0):
        '''
        Initializes the VideoExtractor.

        Parameters:
        env: Gym environment object, expected to be in 'rgb_array' render mode.
        video_folder: str, the path to the folder where generated videos will be saved.
        frame_rate: int, the frame rate for the video. Defaults to 50.
        run_speed_factor: float, a factor to adjust the speed of the video. Defaults to 1.0.
        '''

        if env.render_mode != 'rgb_array':
            raise ValueError("Environment render mode must be 'rgb_array'")

        self.env = env
        self.frame_rate = frame_rate
        self.run_speed_factor = run_speed_factor
        self.video_folder = video_folder

    def generate_video(self, frames: List[np.ndarray], file_name: str = "trajectory", add_timestamp: bool = True) -> str:
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

        file_name = f"{self.video_folder}/{file_name}"

        if add_timestamp:
            file_name = f"{file_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.webm"
        
        fps = self.frame_rate * self.run_speed_factor

        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        video = cv2.VideoWriter(file_name, fourcc, fps, (width, height))

        for frame in frames:
            video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        video.release()

        return file_name
    
    def generate_video_from_pair(self, trajectory_pair: TrajectoryPair):
        """
        Generates videos for a pair of trajectories.

        Parameters:
        trajectory_pair: TrajectoryPair, a TrajectoryPair object.

        Returns:
        tuple: A tuple containing file paths for the two generated videos.
        """
        # Add assertions to ensure trajectory_pair is not modified
        original_t1 = trajectory_pair.trajectory1
        original_t2 = trajectory_pair.trajectory2

        t1 = trajectory_pair.trajectory1
        t2 = trajectory_pair.trajectory2

        frames1 = self.recreate_frames_from_trajectory(t1)
        frames2 = self.recreate_frames_from_trajectory(t2)

        file1 = self.generate_video(frames1)
        time.sleep(1)  # needed to ensure flawless video generation
        file2 = self.generate_video(frames2)

        # Add assertions after video generation
        assert original_t1 == trajectory_pair.trajectory1, "trajectory1 in TrajectoryPair was modified"
        assert original_t2 == trajectory_pair.trajectory2, "trajectory2 in TrajectoryPair was modified"

        return file1, file2

    def recreate_frames_from_trajectory(self, trajectory: Trajectory) -> List[np.ndarray]:
        '''
        Recreates frames from a given trajectory.

        Parameters:
        trajectory: Trajectory, a trajectory object.

        Returns:
        List[np.ndarray]: A list of frames recreated from the trajectory.
        '''
        initial_condition, transitions = trajectory
        if not initial_condition:
            return self._recreate_frames_from_trajectory_no_seed(trajectory)
        else:
            return self._recreate_frames_from_trajectory_with_seed(trajectory)

    def _recreate_frames_from_trajectory_with_seed(self, trajectory: Trajectory) -> List[np.ndarray]:
        initial_condition, transitions = trajectory
        seeds = initial_condition['seed']
        frames = []
        self.env.reset(seed=seeds[0])
        ctr = 0
        for state, action, _, terminated, truncated, _ in transitions:
            print(action.shape)
            self.env.step(action) 
            frames.append(self.env.render())

            if terminated or truncated:
                ctr += 1
                seed = seeds[ctr]
                self.env.reset(seed=seed)

        return frames

    def _recreate_frames_from_trajectory_no_seed(self, trajectory: Trajectory) -> List[np.ndarray]:
        
        _, transitions = trajectory
        frames = []
        starting_state = transitions[0][0]
        obs, info = self.env.reset(obs=starting_state)
        
        for _, action, _, terminated, truncated, next_obs in transitions:
            if terminated or truncated:
                self.env.reset(obs=next_obs)
            else:
                self.env.step(action) 
            frames.append(self.env.render())

        return frames
    
    