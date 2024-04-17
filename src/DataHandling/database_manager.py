import subprocess
import numpy as np
import os
import shutil
import yaml

from abc import ABC, abstractmethod
from pymongo import MongoClient, errors
from typing import Tuple, Optional, List

from src.DataHandling.trajectory_pair import Transition, Trajectory, TrajectoryPair, from_bson, compare_trajectories

CONFIG_PATH = '../../config.yaml'
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)

DB_FOLDER = config["DATABASE_FOLDER"]

class DBManager(ABC):

    def __init__(self, Debug=True):
        pass
    
    @abstractmethod
    def start_db(self):
        pass

    @abstractmethod
    def close_db(self):
        pass

    @abstractmethod
    def add_entry(self):
        pass
    
    @abstractmethod
    def update_entry(self):
        pass

    #@abstractmethod
    #def delete_entry(self):
    #    pass

class MongoDBManager(DBManager):
    """
    Manages database operations for storing and retrieving trajectory pairs and their associated data.

    Attributes:
        client (MongoClient): The MongoDB client for database operations.
        db: The MongoDB database instance.
        collection: The MongoDB collection for storing video data.
        id_of_current_video: The ID of the current video being processed (for sequential access).

    Methods:
        add_entry: Adds a new trajectory pair entry to the database.
        fetch_entry: Retrieves a specific trajectory pair entry from the database.
        set_preference: Updates the preference value of a specific trajectory pair entry.
        skip_pair: Marks a specific trajectory pair entry as skipped.
        get_next_entry: Retrieves the next trajectory pair entry that has not been processed.
    """

    def __init__(self, debug=True):
        """
        Initializes the DBManager with a MongoDB client and sets up the database and collection.
        """
        self.debug = debug
        self.mongod_process = self.start_db()
        self.client = MongoClient("localhost", 27017)
        self.db = self.client['database']
        self.collection = self.db.videos
        self.id_of_current_video = None

    def start_db(self):
        return subprocess.Popen(
        ["mongod", "--port", "27017", "--dbpath", DB_FOLDER],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    def add_entry(self, trajectory_pair: TrajectoryPair) -> Tuple[Optional[str], Optional[str]]:
        """
        Adds a new entry to the database based on the provided trajectory pair.

        Args:
            trajectory_pair (TrajectoryPair): The trajectory pair data to be stored.

        Returns:
            Tuple[Optional[str], Optional[str]]: The inserted ID if successful, and an error message if not.
        """
        try:
            entry = trajectory_pair.to_bson()
            inserted_id = self.collection.insert_one(entry).inserted_id
            
            assert(self.findPair(trajectory_pair) == trajectory_pair)

            return f"Added {str(inserted_id)} to the database", None
        except errors.ConnectionFailure:
            return None, "Connection to DB could not be made."
        except Exception as e:
            return None, f"Not Added successfully to db:{e}"
    
    def gather_results(self) -> List[Optional[float]]:
        """
        Retrieves the 'preference' attribute from each entry in the database.

        Returns:
            List[Optional[float]]: A list of preference values from all entries in the database.
        """
        try:
            documents = self.collection.find()

            preferences = [doc.get('preference', None) for doc in documents]
            
            return preferences
        except errors.ConnectionFailure:
            print("Connection to DB could not be made.")
            return []
        except Exception as e:
            print(f"Error gathering results from db: {e}")
            return []

    #TODO: Test out + edit entry in DB
    def trim_pair(self, pair: TrajectoryPair, 
                video1_start: float, video1_end: float, 
                video2_start: float, video2_end: float) -> TrajectoryPair:
        """
        Trims the trajectories in the TrajectoryPair based on the specified frame ranges.

        Args:
            pair (TrajectoryPair): The trajectory pair to be trimmed.
            video1_start, video1_end (float): Frame range for the first trajectory.
            video2_start, video2_end (float): Frame range for the second trajectory.

        Returns:
            TrajectoryPair: A new TrajectoryPair with trimmed trajectories.
        """
        if video1_start is not None and video1_end is not None:
            trimmed_trajectory1 = self.trim_trajectory(pair.trajectory1, video1_start, video1_end)
        if video2_start is not None and video2_end is not None:
            trimmed_trajectory2 = self.trim_trajectory(pair.trajectory2, video2_start, video2_end)

        return TrajectoryPair(trimmed_trajectory1, trimmed_trajectory2, pair.preference).to_bson

    def trim_trajectory(self, trajectory: Trajectory, start: float, end: float) -> Trajectory:
        """
        Helper method to trim a trajectory based on frame range.

        Args:
            trajectory (Trajectory): The trajectory to be trimmed.
            start, end (float): The frame range for trimming.

        Returns:
            Trajectory: The trimmed trajectory.
        """
        # Initialize variables
        new_transitions = []
        new_seeds = []
        frame_counter = 0
        seed_counter = 0

        # Iterate through transitions and trim
        for transition in trajectory.transitions:
            if start <= frame_counter <= end:
                new_transitions.append(transition)
            if transition.terminated or transition.truncated:
                frame_counter += 1
                seed_counter += 1
                if start <= frame_counter <= end:
                    new_seeds.append(trajectory.initial_conditions['seed'][seed_counter])

        # Update initial conditions
        new_initial_conditions = trajectory.initial_conditions.copy()
        new_initial_conditions['seed'] = new_seeds

        return Trajectory(new_initial_conditions, new_transitions)

    def findPair(self, trajectory_pair: TrajectoryPair) -> TrajectoryPair:
        """
        Finds a trajectory pair entry in the database.

        Args:
            trajectory_pair (TrajectoryPair): The trajectory pair to find.

        Returns:
            dict: The found entry or None if no entry is found.
        """
        trajectory_pair_dict = trajectory_pair.to_bson()
        query = {
            'trajectory1': trajectory_pair_dict["trajectory1"],
            'trajectory2': trajectory_pair_dict["trajectory2"]
        }
        #print(trajectory_pair_dict["trajectory1"])
        #print(trajectory_pair_dict["trajectory2"])
        entry = self.collection.find_one(query)

        if entry is None:
            print("No entry found")
            return None

        assert(trajectory_pair == from_bson(entry))
        
        return from_bson(entry)

    def update_entry(self, trajectory_pair: TrajectoryPair) -> Tuple[Optional[str], Optional[str]]:
        """
        Updates a trajectory pair entry in the database with new preference or skipped status.

        Args:
            trajectory_pair (TrajectoryPair): The trajectory pair to update.

        Returns:
            Tuple[Optional[str], Optional[str]]: Success message or error message.
        """
        entry = self.findPair(trajectory_pair)
        if not entry:
            return None, "No entry found matching the criteria."
        entry = entry.to_bson()

        update_fields = {}
        if hasattr(trajectory_pair, 'preference'):
            update_fields['preference'] = trajectory_pair.preference
        if hasattr(trajectory_pair, 'skipped'):
            update_fields['skipped'] = trajectory_pair.skipped

        if not update_fields:
            return None, "No update fields provided."

        filter_criteria = {
            'trajectory1': entry["trajectory1"],
            'trajectory2': entry["trajectory2"]
        }

        new_values = {"$set": update_fields}

        updated_result = self.collection.update_one(filter_criteria, new_values)

        if updated_result.modified_count == 0:
            return None, "Entry found but no update was needed."
        else:
            return "Entry successfully updated.", None

    def get_next_entry(self):
        try:
            base_query = {"preference": None, "skipped": False}

            # If current_id is set, modify the query to start after that ID
            if self.id_of_current_video is not None:
                base_query["_id"] = {"$gt": self.id_of_current_video}

            entry = self.collection.find_one(base_query, sort=[('_id', 1)])

            if entry:
                self.id_of_current_video = entry["_id"]
                trajectory_pair = from_bson(entry)
                return trajectory_pair, None
            else:
                return None, "No more unprocessed entries found."
        except errors.ConnectionFailure:
            return None, "Connection to DB could not be made."
        except Exception as e:
            return None, str(e)

    def close_db(self):
        self.mongod_process.terminate()
        self.mongod_process.wait()
        if self.debug:
            db_directory = os.path.join(os.path.dirname(__file__),'..', '..', 'data', 'db')
            for filename in os.listdir(db_directory):
                file_path = os.path.join(db_directory, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        print("DB Closed")
