# MiniDRL 

Minimal implementations of distributed deep reinforcement learning algorithms, with a focus on recurrent neural networks. Heavily inspired by [CleanRL](https://github.com/vwxyzjn/cleanrl) and [CORL](https://github.com/corl-team/CORL) this library provides high-quality and easy-to-follow stand-alone implementations of some distributed RL algorithms.

## Getting Started
    
Prerequisites:

- python >= 3.9  (tested with 3.10)

To install:

```bash
git clone git@github.com:Jjschwartz/miniDRL.git
cd miniDRL
pip install -e .
# or to install all dependencies
pip install -e .[all]
```

Run PPO on [gymnasium](https://gymnasium.farama.org/) ``CartPole-v1`` environment using four parallel workers (reduce number of workers if you have less than four cores, or feel free to increase it if you have more):

```bash
python minidrl/ppo/run_gym.py \
    --env_id CartPole-v1 \
    --total_timesteps 1000000 \
    --num_workers 4

# open another terminal and run tensorboard from repo root directory
tensorboard --logdir runs
```

To use experiment tracking with wandb, run:

```bash
wandb login  # only required for the first time
python minidrl/ppo/run_gym.py \
    --env_id CartPole-v1 \
    --total_timesteps 1000000 \
    --num_workers 4 \
    --track_wandb \
    --wandb_project minidrltest
```

## Algorithms

This repository contains standalone implementations of some of the main distributed RL algorithms that support recurrent neural networks, including:

### PPO - Single Machine

[Paper](https://arxiv.org/abs/1707.06347) | [code](https://github.com/Jjschwartz/miniDRL/blob/main/minidrl/ppo/ppo.py) | [docs](https://github.com/Jjschwartz/miniDRL/blob/main/docs/ppo/ppo.md)

|![Learning Curve by wall time vs num workers](docs/ppo/figures/pong_vs_num_workers_wall_time.svg)|
|:--:|
|*Learning curve of PPO - Single Machine on Atari Pong with different number of parallel workers*|

### PPO - Multi Machine

**Coming Soon**

### R2D2

**Coming Soon**

[Paper](https://openreview.net/forum?id=r1lyTjAqYX) | [code](https://github.com/Jjschwartz/miniDRL/tree/main/minidrl/r2d2) | [docs]()



