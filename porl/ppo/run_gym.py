"""Run PPO on classic gym environments."""

from porl.ppo.ppo import run_ppo
from porl.ppo.utils import parse_ppo_args

if __name__ == "__main__":
    run_ppo(parse_ppo_args())
