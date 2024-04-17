from src.DataHandling.trajectory_pair import Trajectory, TrajectoryPair
from src.DataHandling.simulator import Simulator
from src.DataHandling.database_manager import DBManager

import numpy as np
class VideoStreamer():

    def __init__(self, db_manager: DBManager, simulator: Simulator, seed:int = 42):
        self.db_manager = db_manager
        self.simulator = simulator
        self.seed_generator = self._create_seed_generator(seed)

    def _create_seed_generator(self, seed):
        rng = np.random.default_rng(seed)
        while True:
            yield rng.integers(low=0, high=10000)
    
    def stream_env(self, n):
        for _ in range(n):
            seed1 = next(self.seed_generator)
            seed2 = next(self.seed_generator)
            frames1, trajectory1 = self.simulator.simulate_for_n_seconds(seconds=3, seed=seed1)
            frames2, trajectory2 = self.simulator.simulate_for_n_seconds(seconds=3, seed=seed2)
            print(self.db_manager.add_entry(TrajectoryPair(trajectory1, trajectory2)))
        