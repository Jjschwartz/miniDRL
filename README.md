# Partially Observable RL

Implementations of Deep Reinforcement Learning algorithms for partially observable environments using PyTorch.

## Installation
    
Pre-requisites:

- python >= 3.8
- poetry >= 1.1.4

```bash
git clone git@github.com:Jjschwartz/po-rl.git
cd po-rl
poetry install
```

## Algorithms

This repository contains implementations of distributed RL algorithms for partially observable environments implemented using PyTorch. Included are implementations the following algorithms:

- [PPO](https://arxiv.org/abs/1707.06347)
- [R2D2](https://openreview.net/forum?id=r1lyTjAqYX)

### PPO

PORL comes with an implementation of the PPO algorithm with and LSTM based network. The implementation is based on the [CleanRL](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy) implementation of PPO. The PORL implementations includes support for running multiple rollout workers, which collect experience in parallel and send it to the learner. The learner then updates the policy network based on the collected experience.

To run PPO on [gymnasium](https://gymnasium.farama.org/) environments:

```bash
# get help
poetry run python porl/ppo/run_gym.py --help
# run with defaults
poetry run python porl/ppo/run_gym.py
```

Alternatively, to run on [Minigrid](https://github.com/Farama-Foundation/MiniGrid) environments:

```bash
poetry run python porl/ppo/run_minigrid.py
```

### R2D2


