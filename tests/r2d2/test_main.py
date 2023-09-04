"""Tests for R2D2 main function."""
from minidrl.r2d2.r2d2 import run_r2d2
from minidrl.r2d2.utils import R2D2Config


def test_run_r2d2():
    """Tests running R2D2."""
    env_id = "CartPole-v1"
    config = R2D2Config(
        env_id=env_id,
        replay_buffer_size=32,
        num_actors=1,
        num_envs_per_actor=1,
        total_timesteps=80,
        batch_size=8,
        seq_len=8,
        burnin_len=4,
        trunk_sizes=[8],
        lstm_size=8,
        head_sizes=[8],
        debug_actor_steps=None,
        debug_no_logging=True,
    )
    run_r2d2(config)


if __name__ == "__main__":
    test_run_r2d2()
