import queue
import threading
import time
import logging

from typing import Optional

from rlduels.src.primitives.trajectory_pair import TrajectoryPair
from rlduels.src.database.database_manager import DBManager
from rlduels.src.create_video import create_videos_from_pair

class BufferedQueueManager:
    """
    Manages a buffered queue that asynchronously retrieves and stores trajectory pairs from a database.

    This class creates a queue that is continuously refilled in a separate thread. It retrieves
    trajectory pairs from a database, creates videos of trajectories on demand, and stores them
    in the queue for later retrieval.
    """

    def __init__(self, db_manager: DBManager, n: int = 10, sleep_interval: float = 1.0, daemon: bool = True):
        """
        Initializes the BufferedQueueManager with a database manager, queue size, and sleep interval.

        Args:
            db_manager (DBManager): An instance of DBManager for database operations.
            n (int): The maximum size of the buffered queue.
            sleep_interval (int): The interval between queue refills in seconds.
            daemon (bool): Whether the refilling thread should run as a daemon.
        """        
        self.buffered_queue = queue.Queue(maxsize=n)
        self.db_manager: DBManager = db_manager
        self.sleep_interval = sleep_interval
        self._is_running = threading.Event()
        self._is_running.set()

        self.last_processed_id: Optional[str] = None

        self.refilling_thread = threading.Thread(target=self._refill_loop)
        self.refilling_thread.daemon = daemon
        self.refilling_thread.start()

    def _refill_loop(self):
        """
        Continuously refills the queue with new entries. Runs as a separate thread.
        """
        while self._is_running.is_set():
            try:
                if self.get_queue_size() < self.get_queue_max():
                    new_entry, error = self.db_manager.get_next_entry()
                    if not error and new_entry:
                        create_videos_from_pair(new_entry)
                        self.buffered_queue.put(new_entry)
                        self.last_processed_id = new_entry.id
                    else:
                        logging.info(f"Couldn't fetch entry: {error}")
            except Exception as e:
                logging.error(f"Error in refill loop: {e}")
            time.sleep(self.sleep_interval)

    def get_next_entry(self) -> TrajectoryPair:
        """
        Retrieves the next entry from the queue.

        Returns:
            TrajectoryPair: The next entry in the queue.
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
        return self._is_running.is_set()
    
    def close_routine(self):
        """
        Stops the refilling loop and waits for the thread to finish.
        """
        self._is_running.clear()
        logging.info("Closing queue.")
        self.refilling_thread.join()
