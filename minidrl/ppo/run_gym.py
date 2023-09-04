"""Run PPO on classic gym environments."""

from minidrl.ppo.ppo import run_ppo
from minidrl.ppo.utils import parse_ppo_args

if __name__ == "__main__":
    run_ppo(parse_ppo_args())
