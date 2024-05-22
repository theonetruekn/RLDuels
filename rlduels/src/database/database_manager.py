from abc import ABC, abstractmethod
import json
import logging
import os
import shutil
import subprocess
import time
from typing import List, Optional, Tuple

import numpy as np
from pymongo import MongoClient, errors
import yaml

from rlduels.src.primitives.trajectory_pair import (
    TrajectoryPair,
)

CONFIG_PATH = 'config.yaml'
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)

DB_FOLDER = config["DATABASE_FOLDER"]

class DBManager(ABC):

    def __init__(self, debug=True):
        pass
    
    @abstractmethod
    def start_db(self):
        pass

    @abstractmethod
    def close_db(self):
        pass

    @abstractmethod
    def add_entry(self, trajectory_pair: TrajectoryPair) -> Tuple[Optional[str], Optional[str]]:
        pass
    
    @abstractmethod
    def update_entry(self, trajectory_pair: TrajectoryPair) -> Tuple[Optional[str], Optional[str]]:
        pass

    @abstractmethod
    def delete_entry(self, trajectory_pair: TrajectoryPair) -> Tuple[Optional[str], Optional[str]]:
        pass

    @abstractmethod
    def get_next_entry(self) -> Tuple[Optional[TrajectoryPair], Optional[str]]:
        pass

    @abstractmethod
    def gather_preferences(self) -> List[Tuple[Optional[TrajectoryPair], Optional[float]]]:
        pass

class MongoDBManager(DBManager):

    def __init__(self, debug=True, client=MongoClient("localhost", 27017)):
        """
        Initializes the DBManager with a MongoDB client and sets up the database and collection.
        """
        self.debug = debug
        self.mongod_process = self.start_db()
        self.client = client
        self.db = self.client['database']
        self.collection = self.db.videos
        self.id_of_current_video = None

    def start_db(self):
        try:
            process = subprocess.Popen(
                ["mongod", "--port", "27017", "--dbpath", DB_FOLDER],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            time.sleep(0.5)
            if process.poll() is not None:
                stderr = process.stderr.read().decode()
                logging.error("MongoDB failed to start: %s", stderr)
                return None
            return process
        except Exception as e:
            logging.exception("An error occurred while attempting to start MongoDB: %s", e)
            return None

    def add_entry(self, trajectory_pair: TrajectoryPair) -> Tuple[Optional[str], Optional[str]]:
        try:
            serialized_data = trajectory_pair.json()

            data_to_store = json.loads(serialized_data)

            data_to_store['_id'] = data_to_store.pop('id')

            self.collection.insert_one(data_to_store)
            return f"Added {data_to_store['_id']} to the database", None
        except errors.ConnectionFailure:
            return None, "Connection to DB could not be made."
        except Exception as e:
            return None, f"Not added successfully to db: {e}"

    def delete_entry(self, trajectory_pair: TrajectoryPair) -> Tuple[Optional[str], Optional[str]]:
        """
        Deletes a trajectory pair entry in the database using its UUID.

        Args:
            trajectory_pair (TrajectoryPair): The trajectory pair whose entry is to be deleted.

        Returns:
            Tuple[Optional[str], Optional[str]]: Success message or error message.
        """
        try:
            trajectory_id = str(trajectory_pair.id)

            result = self.collection.delete_one({'_id': trajectory_id})

            if result.deleted_count > 0:
                return f"Entry with ID {trajectory_id} successfully deleted.", None
            else:
                return None, f"No entry found with ID {trajectory_id}."

        except errors.ConnectionFailure:
            return None, "Connection to DB could not be made."
        except Exception as e:
            return None, f"Error deleting the entry: {str(e)}"
    
    def find_entry(self, trajectory_pair: TrajectoryPair) -> Optional[TrajectoryPair]:
        """
        Finds a trajectory pair entry in the database using its UUID.

        Args:
            trajectory_pair (TrajectoryPair): The trajectory pair to find based on its UUID.

        Returns:
            Optional[TrajectoryPair]: The found entry or None if no entry is found.
        """
        id = str(trajectory_pair.id)
        try:
            query = {'_id': id}
            entry = self.collection.find_one(query)

            if entry is None:
                logging.error(f"No entry found for object with id {id}.")
                return None
            entry['id'] = entry.pop('_id')
            return TrajectoryPair.from_db(entry)
        except Exception as e:
            logging.error(f"An error occurred parsing the object: {e}")
            return None

    def update_entry(self, trajectory_pair: TrajectoryPair) -> Tuple[Optional[str], Optional[str]]:
        """
        Updates a trajectory pair entry in the database based on its UUID, with new preference or skipped status.

        Args:
            trajectory_pair (TrajectoryPair): The trajectory pair to update.

        Returns:
            Tuple[Optional[str], Optional[str]]: Success message or error message.
        """
        filter_criteria = {'_id': str(trajectory_pair.id)}

        update_fields = {}
        if trajectory_pair.preference is not None:
            update_fields['preference'] = trajectory_pair.preference
        if trajectory_pair.skipped is not None:
            update_fields['skipped'] = trajectory_pair.skipped

        if not update_fields:
            return None, "No update fields provided."

        new_values = {"$set": update_fields}
        try:
            updated_result = self.collection.update_one(filter_criteria, new_values)

            if updated_result.modified_count == 0:
                return None, "Entry found but no update was needed."
            else:
                return f"Entry successfully updated. {updated_result.modified_count} documents affected.", None
        except Exception as e:
            return None, f"Error updating the entry: {str(e)}"

    def get_next_entry(self) -> Tuple[Optional[TrajectoryPair], Optional[str]]:
        """
        Fetches the next unprocessed trajectory pair from the database.

        Returns:
            Tuple[Optional[TrajectoryPair], Optional[str]]: The next trajectory pair if available, and None or error message otherwise.
        """
        try:
            logging.debug("Getting next entry.")
            base_query = {"preference": None, "skipped": False}

            if self.id_of_current_video is not None:
                base_query["_id"] = {"$gt": self.id_of_current_video}

            entry = self.collection.find_one(base_query, sort=[('_id', 1)])
            
            if entry:
                logging.debug("Found Entry in DB.")
                self.id_of_current_video = entry["_id"]

                entry['id'] = entry.pop('_id')
                trajectory_pair = TrajectoryPair.from_db(entry)
                return trajectory_pair, None
            else:
                return None, "No more unprocessed entries found."
        except errors.ConnectionFailure:
            return None, "Connection to DB could not be made."
        except Exception as e:
            return None, f"An error occurred getting the next entry: {str(e)}"

    def gather_preferences(self) -> List[Tuple[Optional[TrajectoryPair], Optional[float]]]:
        """
        Retrieves each entry from the database and collects the 'preference' attribute and the TrajectoryPair.

        Returns:
            List[Tuple[Optional[TrajectoryPair], Optional[float]]]: A list of tuples containing each TrajectoryPair and its preference.
        """
        try:
            documents = self.collection.find()
            results = []

            for doc in documents:
                try:
                    preference = doc.get('preference', None)
                    doc['id'] = doc.pop('_id')
                    trajectory_pair = TrajectoryPair.from_db(doc)
                    results.append((trajectory_pair, preference))
                except Exception as e:
                    logging.error(f"Error processing document {doc['_id']}: {e}")
                    continue
            return results

        except errors.ConnectionFailure:
            logging.error("Connection to DB could not be made.")
            return []
        except Exception as e:
            logging.error(f"Error gathering results from db: {e}")
            return []

    def close_db(self):
        self.mongod_process.terminate()
        self.mongod_process.wait()
        if self.debug:
            logging.debug("Deleting every entry from the database.")
            db_directory = os.path.join(os.path.dirname(__file__),'..', '..', 'data', 'db')
            for filename in os.listdir(db_directory):
                file_path = os.path.join(db_directory, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
        logging.info("Database closed successfully.")
