import gymnasium as gym

from rlduels.src.primitives.trajectory_pair import Transition, Trajectory, TrajectoryPair, NDArray
from rlduels.src.env_wrapper import GymWrapper

def simulate_trajectories(env="CartPole-v1", n=5):
    env = GymWrapper.create_env(name=env, render_mode="rgb_array")

    trajectories = []

    for episode in range(n):
        seed = 42
        observation, info = env.reset(seed=42)
        total_reward = 0
        done = False

        transitions = []

        while not done:
            env.render()

            action = env.sample_action()

            next_observation, reward, terminated, truncated, info = env.step(action)

            transitions.append(Transition.create(
                state=observation,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                next_state=next_observation
            ))

            observation = next_observation
            total_reward += reward
            done = terminated or truncated
        
        trajectories.append(Trajectory(
            env_name = env.name,
            information={'seed': 42},
            transitions=transitions
        ))

        if done:
            print(f"Episode {episode + 1}: Total reward = {total_reward}")

    env.close()

    return trajectories

def simulate_trajectory_pairs(env="CartPole-v1", n=5):
    trajectories = simulate_trajectories(env, 2 * n)
    trajectory_pairs = []

    for i in range(0, len(trajectories), 2):
        trajectory_pair = TrajectoryPair(
            trajectory1=trajectories[i],
            trajectory2=trajectories[i + 1]
        )
        trajectory_pairs.append(trajectory_pair)

    return trajectory_pairs

