import torch as th
from torch.nn.functional import mse_loss
import numpy as np

import gym

import wandb
from typing import Optional, Union, Callable, Final
from utils import *
from models import *
from agents import *
import warnings

warnings.filterwarnings("ignore")

log_t: Final = 100


def lapse(
    args,
    cur_step: int,
    adapt_alpha: float,
    env_info: EnvInfo,
    train_env: gym.Env,
    test_env: gym.Env,
    old_policy: WocarPolicyModel,
    device: th.device,
):
    # get rb
    noise_ball = None
    buf = VanillaOverlapBuffer(device, args.buffer_size)
    if args.fixed_data:
        buf, rew_mean = get_buf(args, train_env, old_policy, args.buffer_size, device, buf)
        if args.policy_smooth:
            obs_mean = buf.get_obs_mean()
            noise_ball = args.smooth_eps * obs_mean
        if args.wandb:
            wandb.log(data={"train/episodes_return": rew_mean}, step=cur_step)
    else:
        assert False, "can't handle obs_mean * smooth_eps"

    recon_angent = CGANAgent(args, env_info, device)
    off_agent = TD3BCAgent(args, env_info, noise_ball, device)

    test_t = args.total_steps // args.test_freq
    last_test_time = -1
    recon_a_loss_ref = []
    bc_a_loss_ref = []

    for t in range(cur_step, cur_step + args.total_steps):
        train_y, train_obs, train_a, train_rew, _, train_next_obs, train_done = buf.sample(args.batch_size)

        G_result, recon_log = recon_angent.train(train_obs, train_y)
        td3_policy_update = t % args.policy_frequency == 0
        off_log = off_agent.train(train_obs, train_a, train_rew, train_next_obs, train_done, adapt_alpha, td3_policy_update)

        if td3_policy_update:
            if t - cur_step > args.total_steps * 0.75:
                with th.no_grad():
                    recon_a = old_policy(G_result)
                    recon_a_loss_info = mse_loss(recon_a, train_a).item()
                recon_a_loss_ref.append(recon_a_loss_info)
                bc_a_loss_ref.append(off_log["train/bc_loss"])

            if t % log_t == 0:
                recon_log.update(off_log)
                log = recon_log
                if args.wandb:
                    wandb.log(data=log, step=t)
                else:
                    log_text = " ".join([k.replace("train/", "") + "=" + f"{v:.4f}" for k, v in log.items()])
                    print(f"[{t}/{args.total_steps}][training] {log_text}")

        if t - last_test_time >= test_t:
            off_returns, off_bc_loss = evaluate(off_agent.policy, test_env, "new", ref_model=old_policy, device=device)
            recon_returns, recon_loss = evaluate(
                old_policy,
                test_env,
                "recon",
                recon_model=recon_angent.recon,
                device=device,
            )
            off_returns, off_bc_loss = np.mean(off_returns), np.mean(off_bc_loss)
            recon_returns, recon_loss = np.mean(recon_returns), np.mean(recon_loss)
            if args.wandb:
                wandb.log(
                    data={
                        "test/recon_episodes_return": recon_returns,
                        "test/recon_loss": recon_loss,
                        "test/off_episodes_return": off_returns,
                        "test/bc_loss": off_bc_loss,
                    },
                    step=t,
                )
            else:
                print(f"[{t}/{args.total_steps}][test] off_returns={off_returns}, bc_loss={off_bc_loss}")
                print(f"[{t}/{args.total_steps}][test] recon_returns={recon_returns}, recon_loss={recon_loss}")
            last_test_time = t

    recon_a_loss_info, bc_a_loss_info = np.mean(recon_a_loss_ref), np.mean(bc_a_loss_ref)

    return recon_angent.recon, off_agent.policy, recon_a_loss_info, bc_a_loss_info
