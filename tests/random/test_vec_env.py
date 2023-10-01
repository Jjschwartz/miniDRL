import argparse
import time

import gymnasium as gym


def make_env(env_id):
    def thunk():
        return gym.make(env_id)

    return thunk


def time_vec_env(env, num_envs, num_steps):
    env.reset()

    start = time.time()
    for _ in range(num_steps):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            env.reset()
    end = time.time()
    env.close()

    print(f"SPS = {(num_steps * num_envs) / (end - start)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="CartPole-v1")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=10000)
    args = parser.parse_args()
    env_id = args.env_id
    num_envs = args.num_envs
    num_steps = args.num_steps

    env_fn = make_env(env_id)
    print("Testing gym.Env")
    time_vec_env(env_fn(), 1, num_steps)

    print("\nTesting gym.vector.SyncVectorEnv")
    envs = gym.vector.SyncVectorEnv([env_fn for _ in range(num_envs)], copy=False)
    time_vec_env(envs, num_envs, num_steps)

    print("\nTesting gym.vector.AsyncVectorEnv")
    envs = gym.vector.AsyncVectorEnv([env_fn for _ in range(num_envs)], copy=False)
    time_vec_env(envs, num_envs, num_steps)
