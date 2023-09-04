# Distributed Recurrent Proximal Policy Optimization

## Overview

Proximal Policy Optimization (PPO) is arguably the most popular deep RL methods out there. It can be used for both discrete and continuous actions, while being relatively robust to hyperparameter choices. It's also relatively simple to parallelize PPO using multiple workers that collect each batch of data in parallel, before sending the data to a central learner.

Original Paper:

- [Schulman et al (2017) Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

Further reading and resources:

- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [CleanRL PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/)


## Distributed Architecture

This implementation of PPO uses a single learner and multiple workers. Each worker runs a copy of the environment and collects a batch of data using the latest version of the policy parameters. The data is then sent to the learner, which performs a gradient update and sends the updated policy parameters back to the workers. The workers then update their local copy of the policy and collect another batch of data.

The following diagram shows a high-level of the distributed architecture:

![Distributed PPO Architecture](figures/distributed_ppo_architecture.svg)

There are a number of important features to note:

### 1. Policy updates and experience collection are synchronized

The learner will wait for all workers to finish collecting a batch before performing a gradient update, and the workers will wait to recieve the latest policy parameters before collecting another batch of data. 

This greatly simplifies the implementation since we don't need to worry about off-policy corrections or stale data. However, it also means that the learner is idle while the workers are collecting data, and the workers are idle while the learner is performing a gradient update. This is not ideal, but it's a reasonable tradeoff for simplicity. Other architectures such as [IMPALA](http://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf) overcome this limitation.

### 2. Rollouts and policy parameters are shared via shared memory

Our implementation of PPO uses the [torch.multiprocessing](https://pytorch.org/docs/stable/multiprocessing.html) library which handles efficiently sharing data between processes using shared memory. This is greatly reduces the communication overhead between the learner and workers since we don't need to serialize and deserialize the data.

### 3. Supports single machine only

This implementation of PPO is relatively simple, and leads to massive speed-ups when running on a single machine with multiple cores. However, it does not support running across multiple machines since it relies on the shared memory of a single machine. This is a reasonable tradeoff for simplicity, but it does mean that we can't scale to hundreds of workers across multiple machines.


## Implementation details

1. [torch.multiprocessing.queues](https://pytorch.org/docs/stable/multiprocessing.html) are used for communication between processes. This handles moving data into shared memory and efficiently sharing the location of this data between processes.
2. In our implementation each rollout worker uses only a single CPU core. This is fine for the environments and models we're using. For larger models you could also assign GPUs to each worker, but would have to be careful about running out of GPU memory.


## Experimental Results

### Scaling

Here we look at how the steps per second scales with number of CPUs used.



