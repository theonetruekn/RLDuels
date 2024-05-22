import pytest
from mongomock import MongoClient
from rlduels.src.database.database_manager import MongoDBManager
from rlduels.src.utils.simulate import simulate_trajectory_pairs
from rlduels.src.primitives.trajectory_pair import TrajectoryPair

@pytest.fixture
def mock_db():
    client = MongoClient()
    db_manager = MongoDBManager(client=client, debug=False)
    yield db_manager
    db_manager.close_db()

@pytest.fixture
def multiple_trajectory_pairs():
    return simulate_trajectory_pairs(n=5)

def test_add_and_find_multiple_entries(mock_db, multiple_trajectory_pairs):
    for tp in multiple_trajectory_pairs:
        success_message, error = mock_db.add_entry(tp)
        print("Success: ", success_message)
        print("Error: ", error)
        assert success_message
        assert error is None
        found_tp = mock_db.find_entry(tp)
        assert found_tp is not None, f"{found_tp}"
        assert str(found_tp.id) == str(tp.id)

def test_delete_multiple_entries(mock_db, multiple_trajectory_pairs):
    for tp in multiple_trajectory_pairs:
        mock_db.add_entry(tp)

    for tp in multiple_trajectory_pairs:
        success_message, error = mock_db.delete_entry(tp)
        assert success_message == f"Entry with ID {str(tp.id)} successfully deleted."
        assert error is None
        assert mock_db.collection.find_one({"_id": str(tp.id)}) is None

def test_update_multiple_entries(mock_db, multiple_trajectory_pairs):
    # Insert multiple entries with initial preferences
    for tp in multiple_trajectory_pairs:
        tp.preference = False  # Set initial preference
        mock_db.add_entry(tp)

    # Update and verify
    for tp in multiple_trajectory_pairs:
        tp.preference = True  # Change preference
        success_message, error = mock_db.update_entry(tp)
        assert success_message.startswith("Entry successfully updated.")
        updated_doc = mock_db.collection.find_one({"_id": str(tp.id)})
        assert updated_doc and updated_doc['preference'] is True

def test_update_entry(mock_db, multiple_trajectory_pairs):
    for tp in multiple_trajectory_pairs:
        mock_db.add_entry(tp)
    
    for idx, tp in enumerate(multiple_trajectory_pairs):
        new_preference = 1 if idx % 2 == 0 else 0
        new_skipped = idx % 2 == 1
        tp.preference = new_preference
        tp.skipped = new_skipped

        success_message, error = mock_db.update_entry(tp)
        assert success_message is not None
        assert error is None

        updated_tp, _ = next((t, p) for t, p in mock_db.gather_preferences() if t.id == tp.id)
        assert updated_tp.preference == new_preference
        assert updated_tp.skipped == new_skipped

def test_gather_preferences(mock_db, multiple_trajectory_pairs):
    for tp in multiple_trajectory_pairs:
        mock_db.add_entry(tp)
    
    uuid_to_preference = {}
    for idx, tp in enumerate(multiple_trajectory_pairs):
        new_preference = 1 if idx % 2 == 0 else 0
        tp.preference = new_preference
        uuid_to_preference[tp.id] = new_preference

        success_message, error = mock_db.update_entry(tp)
        assert success_message is not None
        assert error is None

    results = mock_db.gather_preferences()
    assert len(results) == len(multiple_trajectory_pairs)
    
    for tp, preference in results:
        expected_preference = uuid_to_preference.get(tp.id)
        assert preference == expected_preference, f"Expected {expected_preference}"

def test_get_next_entry(mock_db, multiple_trajectory_pairs):
    added_ids = set()
    for tp in multiple_trajectory_pairs:
        mock_db.add_entry(tp)
        added_ids.add(str(tp.id))
    
    retrieved_ids = set()
    
    for _ in multiple_trajectory_pairs:
        next_tp, error = mock_db.get_next_entry()
        assert next_tp is not None, "Expected an entry but got None"
        assert error is None, f"Expected no error but got: {error}"
        
        retrieved_ids.add(str(next_tp.id))
    
    # Ensure all added entries were retrieved
    assert added_ids == retrieved_ids, f"Expected IDs {added_ids} but got {retrieved_ids}"
    
    # Ensure no more entries are left
    next_tp, error = mock_db.get_next_entry()
    assert next_tp is None, "Expected None but got an entry"
    assert error == "No more unprocessed entries found.", f"Expected 'No more unprocessed entries found.' but got: {error}"

def test_get_next_entry_with_preferences(mock_db, multiple_trajectory_pairs):
    # Insert multiple entries and set preferences for some
    for tp in multiple_trajectory_pairs:
        mock_db.add_entry(tp)

    # Set preference for the first entry
    first_entry = multiple_trajectory_pairs[0]
    first_entry.preference = True
    mock_db.update_entry(first_entry)

    # Track IDs
    added_ids = {str(tp.id) for tp in multiple_trajectory_pairs}
    retrieved_ids = set()

    # Retrieve entries and skip the first one
    for _ in range(1, len(multiple_trajectory_pairs)):
        next_tp, error = mock_db.get_next_entry()
        assert next_tp is not None, "Expected an entry but got None"
        assert error is None, f"Expected no error but got: {error}"
        
        retrieved_ids.add(str(next_tp.id))
    
    # The first entry should not be in retrieved_ids
    added_ids.remove(str(first_entry.id))
    
    # Ensure all added entries (excluding the first one) were retrieved
    assert added_ids == retrieved_ids, f"Expected IDs {added_ids} but got {retrieved_ids}"
    
    # Ensure no more entries are left
    next_tp, error = mock_db.get_next_entry()
    assert next_tp is None, "Expected None but got an entry"
    assert error == "No more unprocessed entries found.", f"Expected 'No more unprocessed entries found.' but got: {error}"

def test_get_next_entry_with_skipped_entries(mock_db, multiple_trajectory_pairs):
    # Insert multiple entries and mark some as skipped
    for tp in multiple_trajectory_pairs:
        mock_db.add_entry(tp)

    # Skip the first entry
    first_entry = multiple_trajectory_pairs[0]
    first_entry.skipped = True
    mock_db.update_entry(first_entry)

    # Track IDs
    added_ids = {str(tp.id) for tp in multiple_trajectory_pairs}
    retrieved_ids = set()

    # Retrieve entries and skip the first one
    for _ in range(1, len(multiple_trajectory_pairs)):
        next_tp, error = mock_db.get_next_entry()
        assert next_tp is not None, "Expected an entry but got None"
        assert error is None, f"Expected no error but got: {error}"
        
        retrieved_ids.add(str(next_tp.id))
    
    # The first entry should not be in retrieved_ids
    added_ids.remove(str(first_entry.id))
    
    # Ensure all added entries (excluding the first one) were retrieved
    assert added_ids == retrieved_ids, f"Expected IDs {added_ids} but got {retrieved_ids}"
    
    # Ensure no more entries are left
    next_tp, error = mock_db.get_next_entry()
    assert next_tp is None, "Expected None but got an entry"
    assert error == "No more unprocessed entries found.", f"Expected 'No more unprocessed entries found.' but got: {error}"