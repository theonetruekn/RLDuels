import gymnasium as gym
from abc import ABC, abstractmethod

class EnvWrapper(ABC):
    def __init__(self, name, render_mode = "rgb_array", **kwargs):
        self._env_name = name
        self.render_mode = render_mode

    @property
    def name(self):
        """Getter for the environment name."""
        return self._env_name

    @classmethod
    @abstractmethod
    def create_env(cls, name, **kwargs):
        """Should be implemented by subclasses to create an environment instance."""
        pass

    @abstractmethod
    def step(self, action):
        pass
    
    @abstractmethod
    def render(self):
        pass
    
    @abstractmethod
    def reset(self, **kwargs):
        pass
    
    @abstractmethod
    def sample_action(self):
        pass

    @abstractmethod
    def close(self):
        pass

class GymWrapper(EnvWrapper):
    def __init__(self, name, render_mode = "rgb_array"):
        super().__init__(name, render_mode)
        self.env = gym.make(name, render_mode=render_mode)
    
    @classmethod
    def create_env(cls, name, render_mode = "rgb_array", **kwargs):
        if name in gym.envs.registry.keys():
            return cls(name, render_mode = render_mode)
        else:
            raise ValueError(f"{name} is not a valid gym environment.")

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def sample_action(self):
        return self.env.action_space.sample()

    def close(self):
        return self.env.close()