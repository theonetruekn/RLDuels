from abc import ABC, abstractmethod
import logging
import requests
import time
import threading
from typing import List

import numpy as np
import yaml

from rlduels.src.primitives.trajectory_pair import (
    Trajectory,
    TrajectoryPair,
    Transition,
)
from rlduels.src.webserver import run_webserver

CONFIG_PATH = '../../config.yaml'

with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)

WEB_SERVER_URL = config["WEBSERVER_URL"]

class PreferenceCollector(ABC):
    
    def __init__(self):
        self.webserver_thread = None
    
    def is_webserver_running(self) -> bool:
        try:
            response = requests.get(f"{WEB_SERVER_URL}/healthcheck")
            return response.status_code == 200
        except ConnectionError:
            return False

    def start_webserver(self, traj_pairs: List[TrajectoryPair]):
        if not self.is_webserver_running():
            self.webserver_thread = threading.Thread(target=run_webserver, args=(traj_pairs,))
            self.webserver_thread.start()
            logging.info("Webserver started.")
            
            while not self.is_webserver_running():
                logging.info("Waiting for the webserver to be ready...")
                time.sleep(1)
        else:
            logging.info("Webserver is already running.")

    def start_collecting(self, **kwargs):
        traj_pairs = self.parse_inputs(**kwargs)

        assert all(isinstance(pair, TrajectoryPair) for pair in traj_pairs), "All elements in traj_pairs must be of type TrajectoryPair"
        
        self.start_webserver(traj_pairs)
        
        while True:
            try:
                response = requests.get(f"{WEB_SERVER_URL}/get_preferences")
                if response.status_code == 200:
                    result = response.json()
                    break
            except ConnectionError:
                logging.info("Waiting for a result to become ready")
                time.sleep(1)

        return result

    @abstractmethod
    def parse_inputs(self, **kwargs) -> List[TrajectoryPair]:
        pass

class AILP_Adapter(PreferenceCollector):
    
    def __init__(self):
        pass

    def _convert(self, pair:tuple, env) -> TrajectoryPair:
        # first and second are now segment size x obs_space size + action_space size (for 3-DoF-Hit-v0 = 18) 
        first, second = pair[0], pair[1]
        assert first.shape[0] == second.shape[0]
        trans1 = []
        trans2 = []

        obs_space_size = env.get_obs_space_shape()[0]
        action_space_shape = env.get_action_space_shape()
        
        for i in range(first.shape[0]):
            state1 = first[i][:obs_space_size]
            state2 = second[i][:obs_space_size]
            next_state1 = first[i + 1][:obs_space_size] if i < first.shape[0] - 1 else np.full(obs_space_size, np.nan)
            next_state2 = second[i + 1][:obs_space_size] if i < second.shape[0] - 1 else np.full(obs_space_size, np.nan)
            action1 = first[i][obs_space_size:].reshape(action_space_shape)
            action2 = second[i][obs_space_size:].reshape(action_space_shape)

            # Create individual Transition instances
            trans1.append(Transition(state1, action1, 0, False, False, next_state1))
            trans2.append(Transition(state2, action2, 0, False, False, next_state2))

        # Initial Conditions None, as we recreate from Obs directly
        traj1 = Trajectory({}, trans1)
        traj2 = Trajectory({}, trans2)

        return TrajectoryPair(traj1, traj2)
    
    def _create_pairs(self, option1, option2):
        """
        Creates a list of pairs from two input nparrays.

        :param option1: First nparray of shape n x m x o.
        :param option2: Second nparray of shape n x m x o.
        :return: List of nparray pairs, each of shape 2 x m x o.
        """
        n, m, o = option1.shape
        pairs = [np.array([option1[i], option2[i]]) for i in range(n)]  # List comprehension to create pairs
        return pairs


    def parse_inputs(self, **kwargs):
        """
        Parses inputs and converts them into TrajectoryPairs.

        :param **kwargs: Arbitrary keyword arguments containing 'option1', 'option2', and 'env'.
        :return: List of TrajectoryPairs.
        """
        option1 = kwargs.get('option1', None)
        option2 = kwargs.get('option2', None)
        env = kwargs.get('env', None)
        assert option1 is not None, "option1 must not be None"
        assert option2 is not None, "option2 must not be None"
        assert env is not None, "env must not be None"

        pairs = self._create_pairs(option1, option2)
        trajectory_pairs = [self._convert(pair, env) for pair in pairs]
        
        return trajectory_pairs