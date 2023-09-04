"""Tests for R2D2 main function."""
from minidrl.r2d2.r2d2 import run_r2d2
from minidrl.r2d2.utils import R2D2Config


def run_r2d2_cartpole():
    """Tests running R2D2."""
    env_id = "CartPole-v1"
    config = R2D2Config(
        env_id=env_id,
        # tracking
        track_wandb=True,
        # training config
        total_timesteps=10000,
        num_actors=1,
        num_envs_per_actor=8,
        actor_device="cuda",
        # replay
        seq_len=16,
        burnin_len=8,
        replay_buffer_size=10000,
        # training hyperparameters
        gamma=0.99,
        batch_size=128,
        n_steps=1,
        learning_rate=2.5e-4,
        target_network_update_interval=500,
        learning_starts=2500,
        value_rescaling=True,
        value_rescaling_epsilon=0.001,
        max_grad_norm=0.5,
        # model
        trunk_sizes=[64],
        lstm_size=64,
        head_sizes=[64],
        # debug
        debug_actor_steps=None,
        debug_no_logging=False,
    )
    run_r2d2(config)


if __name__ == "__main__":
    run_r2d2_cartpole()
