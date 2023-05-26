# Distributed Recurrent Reinforcement Learning (D2RL)

Simple implementations of distributed, recurrent neural networks (RNNs) based deep reinforcement learning algorithms.

Algorithms are implemented using [PyTorch](https://pytorch.org/).

## Installation
    
Pre-requisites:

- python >= 3.8
- poetry >= 1.1.4

```bash
git clone git@github.com:Jjschwartz/distributed-recurrent-rl.git
cd distributed-recurrent-rl
poetry install
```

## Algorithms

This repository contains standalone implementations of some of the main distributed recurrent RL algorithms, including:

- [PPO](https://arxiv.org/abs/1707.06347)
- [R2D2](https://openreview.net/forum?id=r1lyTjAqYX)

### PPO

D2RL comes with an implementation of the PPO algorithm with and LSTM based network. The implementation is based on the [CleanRL](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy) implementation of PPO but with support for running multiple actors. Each actor collect experience in parallel which is stored in a shared buffer. The learner then updates the policy network based on the collected experience.

To run PPO on [gymnasium](https://gymnasium.farama.org/) environments:

```bash
# get help
poetry run python d2rl/ppo/run_gym.py --help
# run with defaults
poetry run python d2rl/ppo/run_gym.py
```

Alternatively, to run on [Minigrid](https://github.com/Farama-Foundation/MiniGrid) environments:

```bash
poetry run python d2rl/ppo/run_minigrid.py
```

### R2D2


