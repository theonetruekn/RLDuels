import pytest
import queue
import threading
import time
from mongomock import MongoClient
from rlduels.src.database.database_manager import MongoDBManager
from rlduels.src.utils.simulate import simulate_trajectory_pairs
from rlduels.src.primitives.trajectory_pair import TrajectoryPair
from rlduels.src.buffered_queue_manager import BufferedQueueManager

@pytest.fixture
def mock_db():
    client = MongoClient()
    db_manager = MongoDBManager(client=client, debug=False)
    yield db_manager
    db_manager.close_db()

@pytest.fixture
def multiple_trajectory_pairs():
    return simulate_trajectory_pairs(n=5)

@pytest.fixture
def buffered_queue_manager(mock_db):
    return BufferedQueueManager(db_manager=mock_db, n=3, sleep_interval=0.1, daemon=True)

def test_queue_initialization(buffered_queue_manager):
    assert buffered_queue_manager.get_queue_size() == 0, "Queue should be empty on initialization"
    assert buffered_queue_manager.get_queue_max() == 3, "Queue max size should be 3"

def test_queue_refill_and_retrieve(buffered_queue_manager, mock_db, multiple_trajectory_pairs):
    # Add multiple trajectory pairs to the mock database
    for tp in multiple_trajectory_pairs:
        mock_db.add_entry(tp)

    # Allow some time for the queue to refill
    time.sleep(1)

    retrieved_ids = set()
    for _ in range(len(multiple_trajectory_pairs)):
        next_tp = buffered_queue_manager.get_next_entry()
        time.sleep(0.5)
        assert next_tp is not None, "Expected an entry but got None"
        retrieved_ids.add(str(next_tp.id))

    added_ids = {str(tp.id) for tp in multiple_trajectory_pairs}
    assert added_ids == retrieved_ids, f"Expected IDs {added_ids} but got {retrieved_ids}"

def test_queue_refill_stops_on_close(buffered_queue_manager, mock_db, multiple_trajectory_pairs):
    # Add multiple trajectory pairs to the mock database
    for tp in multiple_trajectory_pairs:
        mock_db.add_entry(tp)

    # Allow some time for the queue to refill
    time.sleep(1)

    # Close the queue manager
    buffered_queue_manager.close_routine()

    # Ensure the refilling thread has stopped
    assert not buffered_queue_manager.is_running(), "Queue manager should be stopped"

    # Ensure no more entries are added after stopping
    current_size = buffered_queue_manager.get_queue_size()
    time.sleep(1)
    assert buffered_queue_manager.get_queue_size() == current_size, "Queue size should not change after stopping"

def test_queue_does_not_overfill(buffered_queue_manager, mock_db, multiple_trajectory_pairs):
    # Add more trajectory pairs than the queue can hold
    for tp in multiple_trajectory_pairs:
        mock_db.add_entry(tp)

    # Allow some time for the queue to refill
    time.sleep(3)

    # The queue size should not exceed its maximum size
    assert buffered_queue_manager.get_queue_size() == buffered_queue_manager.get_queue_max(), \
        "Queue size should not exceed its maximum capacity"

