# MiniDRL 

Minimal implementations of distributed deep reinforcement learning algorithms, with a focus on recurrent neural networks. Heavily inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl), this library provides high-quality and easy-to-follow stand-alone implementations of some distributed RL algorithms.


Algorithms are implemented using [PyTorch](https://pytorch.org/).

## Getting Started
    
Pre-requisites:

- python >= 3.9  (tested with 3.10)

```bash
git clone git@github.com:Jjschwartz/miniDRL.git
cd miniDRL
pip install -e .
# or to install all dependencies
pip install -e .[all]
```

## Algorithms

This repository contains standalone implementations of some of the main distributed recurrent RL algorithms, including:

- [PPO](https://arxiv.org/abs/1707.06347)
- [R2D2](https://openreview.net/forum?id=r1lyTjAqYX)

### PPO

MiniDRL comes with an implementation of the PPO algorithm with and LSTM based network. The implementation is based on the [CleanRL](https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_lstmpy) implementation of PPO but with support for running multiple actors. Each actor collect experience in parallel which is stored in a shared buffer. The learner then updates the policy network based on the collected experience.

To run PPO on [gymnasium](https://gymnasium.farama.org/) environments:

```bash
# get help
poetry run python minidrl/ppo/run_gym.py --help
# run with defaults
poetry run python minidrl/ppo/run_gym.py
```

Alternatively, to run on [Minigrid](https://github.com/Farama-Foundation/MiniGrid) environments:

```bash
poetry run python minidrl/ppo/run_minigrid.py
```

### R2D2


