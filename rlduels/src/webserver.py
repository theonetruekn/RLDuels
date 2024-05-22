import os
import logging
import yaml
import gymnasium as gym
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rlduels.src.database.database_manager import DBManager, MongoDBManager
from rlduels.src.primitives.trajectory_pair import TrajectoryPair
from rlduels.src.buffered_queue_manager import BufferedQueueManager

CONFIG_PATH = 'config.yaml'
with open(CONFIG_PATH, 'r') as config_file:
    config = yaml.safe_load(config_file)

# Constants read from config
VIDEO_FOLDER = config['VIDEO_FOLDER']
FRAME_RATE = config['FRAME_RATE']
RUN_SPEED_FACTOR = config['RUN_SPEED_FACTOR']
MAX_ENTRIES = config['MAX_ENTRIES']
MAX_QUEUE_SIZE = config['MAX_QUEUE_SIZE']
RESULT_FILE = config["RESULT_FILE"]

# Global variables
db_manager: DBManager = None
buffered_queue: BufferedQueueManager = None
current_entry: TrajectoryPair = None

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class PreferenceUpdate(BaseModel):
    preference: str

class Timestamps(BaseModel):
    video1_start: float
    video1_end: float
    video2_start: float
    video2_end: float

def initialize_app(pairs):
    global db_manager, env, video_extractor, buffered_queue

    db_manager = MongoDBManager()

    for pair in pairs:
        logging.info(db_manager.add_entry(pair))

    buffered_queue = BufferedQueueManager(db_manager, n=MAX_QUEUE_SIZE, daemon=True)

@app.get("/")
async def index():
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

    return JSONResponse(content=config_status)

@app.get("/get_current_video_pair")
async def get_current_video_pair():
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
            logging.debug("Getting next entry from the queue")
            if current_entry is not None:
                current_entry.delete_videos()
            current_entry = buffered_queue.get_next_entry()
            logging.debug(current_entry)
        else:
            logging.debug("No Preference was set, returning same videos")

        logging.debug(f"Current Entry: {current_entry.video1}, {current_entry.video2}")

        video_file_name_1 = os.path.basename(current_entry.video1)
        video_file_name_2 = os.path.basename(current_entry.video2)

        return JSONResponse(content={
            "video1": video_file_name_1, 
            "video2": video_file_name_2
        }, status_code=201)
    except IndexError:
        raise HTTPException(status_code=500, detail="The queue is empty")

@app.get("/get_rewards_for_trajectories")
async def display_rewards_for_trajectories():
    global current_entry
    try:
        return JSONResponse(content={
            "reward1": get_reward(current_entry.trajectory1),
            "reward2": get_reward(current_entry.trajectory2)
        }, status_code=201)
    except Exception as e:
        logging.error(f"Error getting the rewards for the trajectories: {e}")
        raise HTTPException(status_code=500, detail="Failed to display rewards.")

@app.post("/update_preference")
async def update_preference(request: PreferenceUpdate):
    global current_entry
    logging.debug("Preference is being updated", current_entry)
    if current_entry is not None:
        preference = request.preference
        if preference == "video1":
            current_entry.prefer_video1()
        elif preference == "video2":
            current_entry.prefer_video2()
        elif preference == "equal":
            current_entry.prefer_no_video()

        logging.debug(db_manager.update_entry(current_entry))

        return JSONResponse(content={"success": True, "message": "Preference updated successfully."}, status_code=200)
    else: 
        raise HTTPException(status_code=500, detail="The queue is empty")

@app.get("/skip_video_pair")
async def skip_pair():
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
        logging.debug(db_manager.update_entry(current_entry))
        return JSONResponse(content={"success": True, "message": "Video pair skipped successfully."}, status_code=200)
    except Exception as e:
        logging.error(f"Error skipping video pair: {e}")
        raise HTTPException(status_code=500, detail="Failed to skip video pair.")

#@app.post("/get_trimmed_timestamps")
#async def get_trimmed_timestamps(timestamps: Timestamps):
#    """
#    Endpoint to receive trimmed timestamps for video pairs, allowing users to specify relevant segments
#    of the trajectory for evaluation. Updates the trajectory entry in the database with the new timestamps.
#
#    Parameters:
#    None, but expects a JSON payload with start and end timestamps for both videos in the pair.
#
#    Returns:
#    JSON response indicating the success of receiving and processing the timestamps.
#    """
#    global current_entry
#
#    video1_start = timestamps.video1_start
#    video1_end = timestamps.video1_end
#    video2_start = timestamps.video2_start
#    video2_end = timestamps.video2_end
#
#    db_manager.edit_entry(current_entry)
#    raise NotImplementedError
#    return JSONResponse(content={"status": "success", "message": "Timestamps received"}, status_code=200)


@app.post("/terminate")
async def terminate():
    global buffered_queue, db_manager, current_entry
    result = db_manager.gather_preferences()
    
    current_entry.delete_videos()
    buffered_queue.close_routine()
    db_manager.close_db()

    return JSONResponse(content={"status": "success", "message": "Terminated WebApp"}, status_code=200)

def run_webserver(pairs=None):
    """
    Starts the FastAPI web server, optionally initializing it with a set of trajectory pairs.
    Initializes the application components and starts the server to listen for incoming requests.

    Parameters:
    - pairs (list of TrajectoryPair, optional): Initial set of trajectory pairs to use for the application. Defaults to None.

    This function does not return as it starts a blocking server loop.
    """
    initialize_app(pairs)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == '__main__':
    run_webserver()
