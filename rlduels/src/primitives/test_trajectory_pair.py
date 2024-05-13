import numpy as np
import pytest
from pydantic import ValidationError
from rlduels.src.primitives.trajectory_pair import NDArray, TrajectoryPair  # Import your NDArray class appropriately

def test_with_valid_np_array():
    np_array = np.array([1, 2, 3])
    ndarray = NDArray(array=np_array)
    assert np.array_equal(ndarray.array, np_array), "The NDArray should contain the original numpy array"

def test_with_convertible_list():
    list_input = [1, 2, 3]
    ndarray = NDArray(array=list_input)
    assert np.array_equal(ndarray.array, np.array(list_input)), "The NDArray should contain the numpy array converted from list"

def test_with_dict_containing_array():
    np_array = np.array([1, 2, 3])
    dict_input = {'array': np_array}
    ndarray = NDArray(array=dict_input)
    assert np.array_equal(ndarray.array, np_array), "The NDArray should contain the numpy array from dictionary"

def test_with_dict_without_array_key():
    dict_input = {'not_array': [1, 2, 3]}
    with pytest.raises(ValueError) as excinfo:
        NDArray(array=dict_input)
    assert "Dictionary must contain an 'array' key with np.ndarray" in str(excinfo.value), "A ValueError should be raised if dictionary does not have 'array' key"

# Run tests via pytest CLI
