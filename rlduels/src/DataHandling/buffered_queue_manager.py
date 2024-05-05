import queue
import threading
import time
import copy

from typing import Tuple

from src.DataHandling.trajectory_pair import Trajectory, TrajectoryPair
from src.DataHandling.database_manager import DBManager
from src.DataHandling.video_extractor import VideoExtractor

class BufferedQueueManager():
    """
    Manages a buffered queue that asynchronously retrieves and stores trajectory pairs from a database.

    This class creates a queue that is continuously refilled in a separate thread. It retrieves
    trajectory pairs from a database using a DBManager, generates videos using a VideoExtractor,
    and stores them in the queue for later retrieval.
    """

    def __init__(self, db_manager: DBManager, video_extractor:VideoExtractor, n: int = 10, sleep_interval: int = 1, daemon = True):
        """
        Initializes the BufferedQueueManager with a database manager, video extractor, queue size, and sleep interval.

        Args:
            db_manager (DBManager): An instance of DBManager for database operations.
            video_extractor (VideoExtractor): An instance of VideoExtractor to generate videos from trajectories.
            n (int): The maximum size of the buffered queue.
            sleep_interval (int): The interval between queue refills in seconds.
        """        
        self.buffered_queue = queue.Queue(maxsize=n)
        self.db_manager: DBManager = db_manager
        self.video_extractor: VideoExtractor = video_extractor

        self.sleep_interval = sleep_interval
        self.run = True

        self.last_processed_id = None

        if daemon:
            self.refilling_thread = threading.Thread(target=self.refill_loop)
            self.refilling_thread.daemon = daemon
            self.refilling_thread.start()
        else:
            self.refill_loop()

    
    def refill_loop(self):
        """
        Continuously refills the queue with new entries. Runs as a separate thread.
        """
        while self.run:
            if self.get_queue_size() < self.get_queue_max():
                new_entry, error = self.db_manager.get_next_entry()
                if new_entry is not None:
                    # for some weird reason, we need to use a copy, else FindPair doesnt work???
                    entry_copy = copy.deepcopy(new_entry)
                    video1, video2 = self.video_extractor.generate_video_from_pair(entry_copy)
                    #video1, video2 = self.video_extractor.generate_video_from_pair(new_entry)
                    new_entry.set_video1(video1)
                    new_entry.set_video2(video2)
                    self.buffered_queue.put(new_entry)
                    self.last_processed_id = self.db_manager.id_of_current_video
                else:
                    print(f"No entry yet available: {error}")
                    time.sleep(self.sleep_interval)
            time.sleep(self.sleep_interval)       

    def get_next_entry(self) -> TrajectoryPair:
        """
        Retrieves the next entry from the queue.

        Returns:
            Tuple(Trajectory, str, str): The next entry of the queue.
        """
        return self.buffered_queue.get(block=True)

    def get_queue_size(self) -> int:
        """
        Returns the current size of the queue.

        Returns:
            int: The number of items currently in the queue.
        """
        return self.buffered_queue.qsize()
    
    def get_queue_max(self) -> int:
        """
        Returns the maximum size of the queue.

        Returns:
            int: The maximum capacity of the queue.
        """
        return self.buffered_queue.maxsize

    def is_running(self) -> bool:
        """
        Checks if the refilling loop is still running.

        Returns:
            bool: True if the refilling thread is alive, False otherwise.
        """
        return self.run
    
    def close_routine(self):
        self.run = False
        print("Closing queue")
        self.refilling_thread.join()
