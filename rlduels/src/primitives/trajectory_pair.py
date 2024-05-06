from pydantic import BaseModel, Field, validator,  root_validator
from typing import Generic, TypeVar, Union, Optional, List
from uuid import UUID, uuid4
from pathlib import Path
import numpy as np
import os

class NDArray(BaseModel):
    array: np.ndarray

    @validator('array', pre=True)
    def convert_to_nparray(cls, v):
        if isinstance(v, np.ndarray):
            return v
        try:
            converted = np.array(v)
            return converted
        except ValueError as e:
            raise ValueError(f"Could not convert input to numpy array: {e}")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda x: x.tolist()  # Still providing a JSON encoder if needed
        }

class Transition(BaseModel):
    state: NDArray
    action: NDArray
    reward: Union[int, float]
    terminated: bool
    truncated: bool
    next_state: NDArray

class Trajectory(BaseModel):
    env_name: str
    information: dict
    transitions: List[Transition]

    def get_reward(self):
        return sum(transition.reward for transition in self.transitions)

class TrajectoryPair(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    trajectory1: Trajectory
    trajectory2: Trajectory
    env_name: str = Optional[None]
    preference: Union[int, float, None] = None
    skipped: bool = False
    video1: Optional[Path] = None
    video2: Optional[Path] = None
  
    @root_validator(pre=True)  # pre=True ensures this runs after individual field validators
    def set_env_name(cls, values):
        trajectory1 = values.get('trajectory1')
        if trajectory1:
            values['env_name'] = trajectory1.env_name
        return values

    @classmethod
    def from_db(cls, data: dict):
        data['id'] = UUID(data['id'])
        if 'video1' in data:
            data['video1'] = Path(data['video1'])
        if 'video2' in data:
            data['video2'] = Path(data['video2'])
        return cls(**data)

    @validator('trajectory2')
    def check_same_environment(cls, v, values):
        if 'trajectory1' in values and v.env_name != values['trajectory1'].env_name:
            raise ValueError("Environment of trajectory1 and trajectory2 must be identical")
        return v

    class Config:
        json_encoders = {
            UUID: lambda uuid: str(uuid),
            Path: lambda path: str(path)
        }

    def prefer_video1(self):
        self.preference = 0

    def prefer_video2(self):
        self.preference = 1

    def prefer_no_video(self):
        self.preference = 0.5

    def skip(self):
        self.skipped = True

    def unskip(self):
        self.skipped = False

    def undo_preference(self):
        self.preference = None

    def set_video1(self, path: Union[str, Path]):
        self.video1 = Path(path)

    def set_video2(self, path: Union[str, Path]):
        self.video2 = Path(path)

    def delete_videos(self):
        for video_path in [self.video1, self.video2]:
            if video_path and video_path.exists():
                try:
                    os.remove(video_path)
                except FileNotFoundError as e:
                    print(f"Couldn't delete: {e}")
                except Exception as e:
                    print(f"Error deleting file {video_path}: {e}")