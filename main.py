import torch as th
from torch import nn
import numpy as np
import random
import pickle
import time

import gym


import os
import argparse
import wandb
from typing import Optional, Union, Callable, Final
from utils import *
from models import *
from overlap import *
import warnings

warnings.filterwarnings("ignore")

log_t: Final = 100


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="lapse")
    parser.add_argument("--task", type=str, default="HalfCheetah-v2")
    parser.add_argument("--trans", type=str, default="physics")
    parser.add_argument("--use-wocar", type=int, default=True)

    parser.add_argument("--overlap-times", type=int, default=5)
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--test-freq", type=int, default=1)

    parser.add_argument("--fixed-data", type=int, default=True)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[64, 64])
    parser.add_argument("--optim", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)

    # config for cGAN
    parser.add_argument("--cgan-l2-lambda", type=float, default=10.0)

    # config for TD3
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--policy-frequency", type=int, default=2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)

    # config for lapse
    parser.add_argument("--alpha-max", type=float, default=2.5)
    parser.add_argument("--adapt-tau", type=float, default=0.5)
    parser.add_argument("--lapse-pruning", type=int, default=False)
    parser.add_argument("--pruning-val", type=float, default=0.2)

    # config for RORL
    parser.add_argument("--policy-smooth", type=int, default=True)
    parser.add_argument("--smooth-eps", type=float, default=0.001)
    parser.add_argument("--smooth-reg", type=float, default=0.1)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save", type=int, default=False)

    # wanDB config
    os.environ["WANDB_API_KEY"] = ""
    parser.add_argument("--wandb", type=int, default=False)
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="Lapse_exp")
    return parser.parse_args()


def train():
    args = get_args()
    device = th.device(args.device)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True

    np.set_printoptions(precision=4)
    th.set_printoptions(precision=4)

    env_info = EnvInfo(args.task, args.use_wocar)
    args.trans_base = env_info.base

    # load pi_0 from pretrained Wocar models
    old_policy = (
        WocarLstmPolicyModel(env_info).to(device)
        if "ant" in args.task.lower() and args.use_wocar
        else WocarPolicyModel(env_info).to(device)
    )
    env_rep = args.task.replace("-v2", "").replace("2d", "").lower()
    ckpt_path = f"pretrain/models/{env_rep}_{'wocar' if args.use_wocar else 'natural'}.pt"
    ckpt = th.load(ckpt_path)
    old_policy.load_state_dict(ckpt, strict=False)

    # init wanDB if needed
    if args.tag:
        tags_tmp += [args.tag]
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.method}_{args.task}_{args.trans}_{'wocar' if args.use_wocar else 'natural'}_{args.seed}",
            entity=args.wandb_entity,
            config=args,
            tags=tags_tmp,
        )

    print(vars(args))

    match args.method:
        case "lapse":
            funcs = TransformFuncs(args.overlap_times, env_info, args.trans_base == "wocar", args.trans)
            lapse_pi = LapsePolicys(args.overlap_times, old_policy, pruning=args.lapse_pruning, pruning_val=args.pruning_val)

            cur_t = 0

            test_env = gym.vector.SyncVectorEnv([make_env(env_info, funcs.get_funcs_pair(1, True))])
            episodic_returns, _ = evaluate(old_policy, test_env, "new", eval_episodes=10, device=device)
            R_max = np.mean(episodic_returns)
            R_last = R_max
            print(f"[0/{args.overlap_times}][overlap] return for pi_0: {R_max}")
            if args.wandb:
                wandb.log({"eval/pi_reward": R_max}, step=cur_t)

            for i in range(args.overlap_times):
                # cal adapt_alpha using current stage i (start from 0)
                # cal rew_scale using R_last and R_max
                adapt_alpha = args.alpha_max * (1 - np.exp(-args.adapt_tau * i))
                rew_scale = min(R_last / R_max, 1.0)
                print(f"[{i+1}/{args.overlap_times}][overlap] adapt_alpha: {adapt_alpha}, rew_scale: {rew_scale}")
                if args.wandb:
                    wandb.log({"train/adapt_alpha": adapt_alpha, "train/rew_scale": rew_scale}, step=cur_t)

                # set up envs
                train_env = gym.vector.SyncVectorEnv([make_env(env_info, funcs.get_funcs_pair(i + 1))])
                test_env = gym.vector.SyncVectorEnv([make_env(env_info, funcs.get_funcs_pair(i + 1))])

                # lapse train -> recon_func, distill_policy
                recon_func, distill_policy, recon_a_loss_info, bc_a_loss_info = lapse(
                    args,
                    cur_t,
                    adapt_alpha,
                    env_info,
                    train_env,
                    test_env,
                    lapse_pi,
                    device,
                )

                # cal kappa w.r.t. current recon and td3bc performance
                kappa = rew_scale * bc_a_loss_info / (recon_a_loss_info + bc_a_loss_info)
                print(f"[{i+1}/{args.overlap_times}][overlap] kappa: {kappa}")

                # evaluate the recon_func and distill_policy -> R_recon, R_off
                episodic_returns, _ = evaluate(
                    lapse_pi, test_env, "recon", recon_model=recon_func, eval_episodes=10, device=device
                )
                R_recon = np.mean(episodic_returns)
                episodic_returns, _ = evaluate(distill_policy, test_env, "new", eval_episodes=10, device=device)
                R_off = np.mean(episodic_returns)
                print(f"[{i + 1}/{args.overlap_times}][overlap] R_recon_{i + 1}: {R_recon}, R_off_{i + 1}: {R_off}")

                # update the pi_n
                lapse_pi.update(kappa, recon_func, distill_policy)
                cur_t += args.total_steps

                # evaluate the pi_n -> R_last
                episodic_returns, _ = evaluate(lapse_pi, test_env, "new", eval_episodes=10, device=device)
                R_last = np.mean(episodic_returns)
                print(f"[{i + 1}/{args.overlap_times}][overlap] return for pi_{i + 1}: {R_last}")
                if args.wandb:
                    wandb.log(
                        {
                            "train/kappa": kappa,
                            "train/ref_recon_bc_loss": recon_a_loss_info,
                            "train/ref_bc_bc_loss": bc_a_loss_info,
                            "eval/pi_reward": R_last,
                            "eval/recon_reward": R_recon,
                            "eval/off_reward": R_off,
                            "train/pruning_stage": lapse_pi.pruning_stage,
                            "train/pruning_off": lapse_pi.pruning_off_cnt,
                        },
                        step=cur_t,
                    )

            if args.save:
                ckpt = {"args": vars(args), "funcs": funcs.func_info, "lapse": lapse_pi}
                folder_path = f"saves/{args.tag}/" + time.strftime("%Y-%m-%d", time.localtime(time.time())) + f"/{args.task}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                with open(f"{folder_path}/{'natural' if args.use_wocar else 'wocar'}_s{args.seed}.bin", "wb") as f:
                    pickle.dump(ckpt, f)
        case _:
            assert False, "unkown method"


if __name__ == "__main__":
    train()
