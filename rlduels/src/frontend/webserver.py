import sys
import os
import signal
import threading
import numpy as np
import shutil
import yaml
import traceback
import subprocess
import gymnasium as gym

from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

from src.DataHandling.database_manager import DBManager, MongoDBManager
from src.DataHandling.trajectory_pair import Transition, Trajectory, TrajectoryPair, get_reward
from src.DataHandling.simulator import Simulator
from src.DataHandling.buffered_queue_manager import BufferedQueueManager
from src.DataHandling.video_extractor import VideoExtractor
from src.DataHandling.video_streamer import VideoStreamer

"""
This script manages a web server created to allow users to interact with a Reinforcement Learning (RL) environment via a WebUI.
Users can select preferences, view and interact with trajectory pairs generated from the RL environment, and provide feedback.

Constants:
    CONFIG_PATH: The file path to the configuration file 'config.yaml' which contains various settings for the application.

Global Variables:
    db_manager (DBManager): Manages database operations, handling user preferences and trajectory pair data.
    simulator (Simulator): Facilitates the generation of trajectory pairs by simulating interactions in the RL environment.
    video_streamer (VideoStreamer): Streams and records trajectory pairs from the RL environment, used when initial pairs are not provided.
    video_extractor (VideoExtractor): Extracts and records videos from given trajectory pairs for user interaction.
    env (gym.Env): An instance of a gym environment used for simulations.
    buffered_queue (BufferedQueueManager): Manages a queue of trajectory pairs and associated videos for user evaluation.
    current_entry: The current trajectory pair being evaluated by the user.

Functions:
    initialize_app(pairs): Initializes the application, setting up the database, simulator, and video handlers.
    index(): The main route for the web server, serving the index page with configuration settings.
    get_next_video_pair(): Endpoint to retrieve the next pair of videos for user evaluation from the buffered queue.
    update_preference(): Endpoint to update the user's preference for the currently viewed trajectory pair.
    skip_pair(): Endpoint to skip the current video pair and fetch the next pair from the queue.
    get_trimmed_timestamps(): Endpoint to receive trimmed timestamps for video pairs, allowing users to specify relevant segments.
    shutdown(): Utility function to shut down the Flask server gracefully.
    terminate(): Endpoint to terminate the web server, saving results and closing database connections.
    run_webserver(pairs): Starts the Flask web server in a separate thread, optionally initializing it with a set of trajectory pairs.

The web server provides a user-friendly interface for interacting with trajectory pairs, enabling users to view, evaluate, and provide feedback on RL-generated trajectories.
"""
CONFIG_PATH = 'config.yaml'
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)

# Constants
ENV = config['ENV_NAME']
VIDEO_FOLDER = config['VIDEO_FOLDER']
FRAME_RATE = config['FRAME_RATE']
RUN_SPEED_FACTOR = config['RUN_SPEED_FACTOR']
MAX_ENTRIES = config['MAX_ENTRIES']
MAX_QUEUE_SIZE = config['MAX_QUEUE_SIZE']
RESULT_FILE = config["RESULT_FILE"]

# Global variables
db_manager = None
simulator = None
video_streamer = None
video_extractor = None
env = None
buffered_queue = None
current_entry = None
result = []


app = Flask(__name__)
CORS(app)

def initialize_app(pairs):
    global db_manager, simulator, video_streamer, env, video_extractor, buffered_queue

    db_manager = MongoDBManager()

    # Populate the database with some entries
    if pairs is None:
        env = gym.make(
            ENV,
            render_mode="rgb_array"
        )
        simulator = Simulator(env = env, agent = None, frame_rate = FRAME_RATE, run_speed_factor = RUN_SPEED_FACTOR)
        video_streamer = VideoStreamer(db_manager=db_manager, simulator=simulator, seed=420)
        video_streamer.stream_env(MAX_ENTRIES)
        env.close()
    else:
        for pair in pairs:
            print(db_manager.add_entry(pair))

    # Start a new env, only used in the background thread on the queue
    video_extractor = VideoExtractor(
        gym.make(
            ENV,
            render_mode="rgb_array"
        ),
        VIDEO_FOLDER,
        FRAME_RATE, 
        RUN_SPEED_FACTOR
    )
    buffered_queue = BufferedQueueManager(db_manager, video_extractor, n=MAX_QUEUE_SIZE, daemon=True)


@app.route('/')
def index():
    """
    Serves the main index page of the web application. Loads configuration settings from a YAML file
    and provides them to the index template for rendering the user interface.

    Parameters:
    None

    Returns:
    Rendered template of the index page with configuration status.
    """
    config_status = {'allowTies': False, 'allowSkipping': False, 'allowEditing': False, 'debugMode': False}  # Initialize with default values
    
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as file:
            config = yaml.safe_load(file)
            config_status = {
                'allowTies': config.get('allowTies', False),
                'allowSkipping': config.get('allowSkipping', False),
                'allowEditing': config.get('allowEditing', False),
                'debugMode' : config.get('debugMode', True)
            }
    else:
        raise Exception("No configuration file exists.")

    return render_template('index.html', config=config_status)


@app.route('/get_current_video_pair')
def get_current_video_pair():
    """
    Endpoint to retrieve the next pair of videos for user evaluation. Fetches the next entry from the
    buffered queue and returns the video file names to the client.

    Parameters:
    None

    Returns:
    JSON response containing the file names of the next pair of videos or an error message if the queue is empty.
    """
    global current_entry
    try:
        if current_entry is None or current_entry.preference is not None:
            print("Getting next entry from the queue")
            if current_entry is not None:
                current_entry.delete_videos()
            current_entry = buffered_queue.get_next_entry()
            print(current_entry)
        else:
            print("No Preference was set, returning same videos")

        print(f"Current Entry: {current_entry.video1}, {current_entry.video2}")

        video_file_name_1 = os.path.basename(current_entry.video1)
        video_file_name_2 = os.path.basename(current_entry.video2)

        return jsonify({
            "video1": video_file_name_1, 
            "video2": video_file_name_2
        }), 201
    except IndexError:
        return jsonify({"message": "The queue is empty"}), 500

@app.route('/get_rewards_for_trajectories')
def display_rewards_for_trajectories():
    global current_entry
    try:
        return jsonify({
            "reward1": get_reward(current_entry.trajectory1),
            "reward2": get_reward(current_entry.trajectory2)
        }), 201
    except Exception as e:
        print(f"Error getting the rewards for the trajectories: {e}")
        return jsonify({"success": False, "message": "Failed to display rewards."}), 500

@app.route('/update_preference', methods=['POST'])
def update_preference():
    global current_entry
    print("Preference is being updated", current_entry)
    if current_entry is not None:
        preference = request.form.get('preference')

        if preference == "video1":
            current_entry.prefer_video1()
        elif preference == "video2":
            current_entry.prefer_video2()
        elif preference == "equal":
            current_entry.prefer_no_video()

        print(db_manager.update_entry(current_entry))

        return jsonify({"success": True, "message": "Preference updated successfully."}), 200
    else: 
        return jsonify({"message": "The queue is empty"}), 500

@app.route('/skip_video_pair')
def skip_pair():
    """
    Endpoint to skip the current video pair and fetch the next pair from the queue. Marks the current trajectory
    pair as skipped in the database.

    Parameters:
    None

    Returns:
    JSON response indicating the success or failure of skipping the video pair.
    """
    global current_entry
    try:
        current_entry.skip()
        print(db_manager.update_entry(current_entry))
        return jsonify({"success": True, "message": "Video pair skipped successfully."}), 200
    except Exception as e:
        print(f"Error skipping video pair: {e}")
        return jsonify({"success": False, "message": "Failed to skip video pair."}), 500

@app.route('/get_trimmed_timestamps', methods=['POST'])
def get_trimmed_timestamps():

    """
    Endpoint to receive trimmed timestamps for video pairs, allowing users to specify relevant segments
    of the trajectory for evaluation. Updates the trajectory entry in the database with the new timestamps.

    Parameters:
    None, but expects a JSON payload with start and end timestamps for both videos in the pair.

    Returns:
    JSON response indicating the success of receiving and processing the timestamps.
    """

    data = request.json

    video1_start = data['video1_start']
    video1_end = data['video1_end']
    video2_start = data['video2_start']
    video2_end = data['video2_end']

    db_manager.edit_entry(current_entry)

    return jsonify({"status": "success", "message": "Timestamps received"}), 200

def shutdown():
    """Shuts down the server from a Flask route."""
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/terminate', methods=['POST'])
def terminate():
    global buffered_queue, db_manager, current_entry
    # Before shutting down, write the results to a text file

    result = db_manager.gather_results()

    if None in result and not config["allowSkipping"]:
        return jsonify({"status": "fail", "message": "Evaluate all pairs."}), 500

    with open(RESULT_FILE, 'w') as file:
        file.write(str(result))
    
    current_entry.delete_videos()
    buffered_queue.close_routine()
    db_manager.close_db()
    result = []

    shutdown()
    return jsonify({"status": "success", "message": "Terminated WebApp"}), 200

def run_webserver(pairs=None):
    """
    Starts the Flask web server, optionally initializing it with a set of trajectory pairs.
    Initializes the application components and starts the server to listen for incoming requests.

    Parameters:
    - pairs (list of TrajectoryPair, optional): Initial set of trajectory pairs to use for the application. Defaults to None.

    This function does not return as it starts a blocking server loop.
    """
    initialize_app(pairs)
    app.run()

if __name__ == '__main__':
    run_webserver()