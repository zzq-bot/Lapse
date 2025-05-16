# Learning to Reuse Policies in State Evolvable Environments

We present **Lapse**, addressing the challenge of state evolvable environments in reinforcement learning with the formulation of SERL. Lapse successfully combines a robust policy and an adaptive policy through state reconstruction and offline learning via policy reuse in two different aspects, thus avoiding the need for extensive trial-and-error during deployment.

![image-20250129212827051](/Users/zzq-bot/Library/Application Support/typora-user-images/image-20250129212827051.png)

## Installation instructions

First, install **Mujoco210** and update the `LD_LIBRARY_PATH` environment variable.

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
```

> We recommend building the environment with **Python version 3.10**. Earlier versions do not support the `match` statement in the code, and updated versions may be incompatible with existing type hints.

Build the environment by running:

```shell
pip install -r requirements.txt
```

The requirement of wandb is optional and serves for a more visual presentation of experimental results.

## Run an experiment

Run an experiment by:

```shell
python main.py --task="Ant-v2"
```

To save checkpoints, you can set the arguments `--save=1`. By setting `--wandb=1` and the environment variable `WANDB_API_KEY`, you can view experiment logs and results through wandb.

Additionally, in the `scripts` folder, we have prepared some scripts for conducting multiple sets of experiments:

```shell
bash scripts/lapse.sh
```

By default, we test only once at each stage. If you want to observe the performance of the policy during training at different stages, you can increase the value of the arguments `--test-freq`.
