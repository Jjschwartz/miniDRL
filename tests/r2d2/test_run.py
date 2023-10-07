"""Tests for R2D2 main function."""
import os

import torch
from minidrl.r2d2.r2d2 import R2D2Config, run_r2d2
from minidrl.r2d2.run_gym import get_gym_env_creator_fn, gym_model_loader


def test_run_r2d2_single_actor():
    """Tests running R2D2."""
    print("\n\nRunning test_run_r2d2_single_actor")
    config = R2D2Config(
        env_id="CartPole-v1",
        # tracking
        track_wandb=False,
        # training config
        total_timesteps=100,
        num_actors=1,
        num_envs_per_actor=8,
        actor_device="cuda",
        # replay
        seq_len=16,
        burnin_len=8,
        replay_buffer_size=1000,
        # training hyperparameters
        gamma=0.99,
        batch_size=128,
        learning_starts=128,
        n_steps=1,
        learning_rate=2.5e-4,
        target_network_update_interval=500,
        value_rescaling=True,
        value_rescaling_epsilon=0.001,
        max_grad_norm=0.5,
        # lstm size for default gym model
        # lstm_size_=128,
    )
    config.env_creator_fn_getter = get_gym_env_creator_fn
    config.model_loader = gym_model_loader
    run_r2d2(config)


def test_run_r2d2_multi_actor():
    """Tests running R2D2."""
    print("\n\nRunning test_run_r2d2_multi_actor")
    num_cpus = os.cpu_count()
    assert num_cpus is not None
    if num_cpus <= 4:
        print("Skipping multi-actor test because insufficient CPUs is available.")
        return

    config = R2D2Config(
        env_id="CartPole-v1",
        # tracking
        track_wandb=False,
        # training config
        total_timesteps=100,
        num_actors=2,
        num_envs_per_actor=8,
        actor_device="cuda",
        # replay
        seq_len=16,
        burnin_len=8,
        replay_buffer_size=1000,
        # training hyperparameters
        gamma=0.99,
        batch_size=128,
        learning_starts=128,
        n_steps=1,
        learning_rate=2.5e-4,
        target_network_update_interval=500,
        value_rescaling=True,
        value_rescaling_epsilon=0.001,
        max_grad_norm=0.5,
        # lstm size for default gym model
        # lstm_size_=128,
    )
    config.env_creator_fn_getter = get_gym_env_creator_fn
    config.model_loader = gym_model_loader
    run_r2d2(config)


if __name__ == "__main__":
    # limit threads per actor to 1
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    test_run_r2d2_single_actor()
    test_run_r2d2_multi_actor()
