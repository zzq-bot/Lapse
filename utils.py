import torch as th
from torch import nn
from torch.nn.functional import mse_loss
import gym
import numpy as np
import random
from tqdm import tqdm
import pickle
from copy import deepcopy as dco
from typing import Any, Optional, Union, Callable, Final, Literal, Tuple
from gym.spaces import Tuple as GymTuple
import wandb


DEBLOG: Final = False


class EnvInfo:
    def __init__(self, env_id: str, use_wocar: bool = False, load_stat: bool = True) -> None:
        self.env_id = env_id
        if load_stat:
            run_stat = pickle.load(open(f"pretrain/envs/{env_id} {'wocar' if use_wocar else 'natural'}.bin", "rb"))
            self.mean = run_stat["mean"]
            self.std = run_stat["std"]
            self.base = run_stat["base"]
        tmp_env = gym.make(env_id)
        self.obs_space = tmp_env.observation_space
        self.act_space = tmp_env.action_space
        self.obs_dim = np.prod(tmp_env.observation_space.shape)
        self.act_dim = np.prod(tmp_env.action_space.shape)
        tmp_env.close()


class TransformFuncs:
    def __init__(
        self,
        overlap_times: int,
        env_info: EnvInfo,
        wocar_base: bool,
        trans: Literal["linear", "physics", "vanilla"],
        load_ckpt: dict = None,
    ):
        self.overlap_times = overlap_times
        if load_ckpt is not None:
            print("load funcs from ckpt")
            self.wocar_mean = load_ckpt["wocar_mean"]
            self.wocar_std = load_ckpt["wocar_std"]
            self.wocar_base = load_ckpt["wocar_base"]
            self.trans = load_ckpt["trans"]
        else:
            self.wocar_mean = env_info.mean
            self.wocar_std = env_info.std
            self.wocar_base = wocar_base
            self.trans = trans
        self.wocar_transform = lambda x: np.clip((x - self.wocar_mean) / (self.wocar_std + 1e-8), -10.0, 10.0)
        self.funcs = [self.wocar_transform]
        self.func_info = {
            "wocar_base": wocar_base,
            "wocar_mean": self.wocar_mean,
            "wocar_std": self.wocar_std,
            "trans": trans,
            "coef": {},
        }
        match self.trans:
            case "linear":
                main_coefs = [_ for _ in range(self.overlap_times)]
                # main_coefs = [_ / 2 + 2 if _ % 2 == 0 else -((_ - 1) / 2 + 2) for _ in main_coefs]
                main_coefs = [_ * 2.0 + 2.0 for _ in main_coefs]
                for i in range(self.overlap_times):
                    if load_ckpt is not None:
                        a = load_ckpt["coef"][i]["a"]
                        b = load_ckpt["coef"][i]["b"]
                    else:
                        a = main_coefs[i] + np.random.rand() - 0.5
                        b = np.random.rand()
                    base_func = self.wocar_transform if self.wocar_base else lambda x: x
                    tmp_func = lambda x, a=a, b=b: a * base_func(x) + b
                    self.funcs.append(tmp_func)
                    self.func_info["coef"][i] = {"a": a, "b": b}
            case "physics":
                from pretrain.envs_config import get_physics_func

                for i in range(self.overlap_times):
                    base_func = self.wocar_transform if self.wocar_base else lambda x: x
                    tmp_func = get_physics_func(env_info.env_id, i + 1)
                    ready_func = lambda x, tmp_func=tmp_func, base_func=base_func: tmp_func(base_func(x))
                    self.funcs.append(ready_func)
            case _:
                assert False, "unkown transform func type"

    def get_funcs_pair(self, stage: int, both_wocar: bool = False) -> Tuple:
        if both_wocar:
            return (self.funcs[0], self.funcs[0])
        return (self.funcs[stage - 1], self.funcs[stage])


class ZFilterClipWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, env_info: EnvInfo, trans_func: Optional[Tuple] = None):
        super().__init__(env)
        assert trans_func, "must have a trans_func"
        self.last_func = trans_func[0]
        self.new_func = trans_func[1]
        self.observation_space = GymTuple((env_info.obs_space, env_info.obs_space))

    def observation(self, obs: np.array) -> GymTuple:
        with th.no_grad():
            last_obs = self.last_func(obs)
            new_obs = self.new_func(obs)
        return (last_obs, new_obs)


def make_env(
    env_info: EnvInfo,
    trans_func: Optional[Tuple] = None,
    idx: Optional[int] = 0,
    capture_video: Optional[bool] = False,
    run_name: Optional[str] = "defalut",
) -> Callable:
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_info.env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_info.env_id)

        env = ZFilterClipWrapper(env, env_info, trans_func)
        return env

    return thunk


class VanillaOverlapBuffer:
    def __init__(self, device: th.device, size: int = 1000000):
        self.device = device
        self.buffer = []
        self.full = False
        self.size = size
        self.pos = 0

    def add(self, stateA, stateB, action, reward, next_stateA, next_stateB, done):
        if not self.full:
            self.buffer.append((stateA, stateB, action, reward, next_stateA, next_stateB, done))
        else:
            self.buffer[self.pos] = (stateA, stateB, action, reward, next_stateA, next_stateB, done)
        self.pos = self.pos + 1
        if self.pos == self.size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size, to_tensor: bool = True):
        stateA, stateB, action, reward, next_stateA, next_stateB, done = zip(*random.sample(self.buffer, batch_size))

        if to_tensor:
            return (
                th.Tensor(np.concatenate(stateA)).to(self.device),
                th.Tensor(np.concatenate(stateB)).to(self.device),
                th.Tensor(np.concatenate(action)).to(self.device),
                th.Tensor(np.concatenate(reward)).to(self.device),
                th.Tensor(np.concatenate(next_stateA)).to(self.device),
                th.Tensor(np.concatenate(next_stateB)).to(self.device),
                th.Tensor(np.concatenate(done)).to(self.device),
            )
        else:
            return (
                np.concatenate(stateA),
                np.concatenate(stateB),
                np.concatenate(action),
                np.concatenate(reward),
                np.concatenate(next_stateA),
                np.concatenate(next_stateB),
                np.concatenate(done),
            )

    def __len__(self):
        return len(self.buffer)

    def get_obs_mean(self):
        _, stateB, *_ = zip(*self.buffer)
        return np.mean(stateB)


class RawReplayBuffer:
    def __init__(self, device: th.device, size: int = 500000):
        self.device = device
        self.buffer = []
        self.full = False
        self.size = size
        self.pos = 0

    def add(self, stateB, action, reward, next_stateB, done):
        if not self.full:
            self.buffer.append((stateB, action, reward, next_stateB, done))
        else:
            self.buffer[self.pos] = (stateB, action, reward, next_stateB, done)
        self.pos = self.pos + 1
        if self.pos == self.size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        stateB, action, reward, next_stateB, done = zip(*random.sample(self.buffer, batch_size))

        return (
            th.Tensor(np.concatenate(stateB)).to(self.device),
            th.Tensor(np.concatenate(action)).to(self.device),
            th.Tensor(np.concatenate(reward)).to(self.device),
            th.Tensor(np.concatenate(next_stateB)).to(self.device),
            th.Tensor(np.concatenate(done)).to(self.device),
        )

    def __len__(self):
        return len(self.buffer)

    def get_obs_mean(self):
        stateB, *_ = zip(*self.buffer)
        return np.mean(stateB)


def get_obs(
    obs: Union[GymTuple, np.array],
    method: Literal["old", "new", "recon"] = "old",
    recon_model: Optional[nn.Module] = None,
    device: th.device = th.device("cuda"),
):
    old_obs, new_obs = obs[0], obs[1]
    match method:
        case "old":
            return old_obs
        case "new":
            return new_obs
        case "recon":
            return recon_model(th.Tensor(new_obs).to(device))
        case _:
            assert 0


def evaluate(
    model: nn.Module,
    envs: gym.Env,
    method: Literal["old", "new", "recon"] = "old",
    recon_model: Optional[nn.Module] = None,
    recon_chain=None,
    eval_episodes: int = 3,
    device: th.device = th.device("cpu"),
    ref_model: Optional[nn.Module] = None,
):
    model.eval()

    episodic_returns = []
    method_loss = []

    for i in range(eval_episodes):
        obs_pair = envs.reset()
        if hasattr(model, "reset"):
            model.reset()
        if ref_model and hasattr(ref_model, "reset"):
            ref_model.reset()
        rew_sum = 0
        while True:
            with th.no_grad():
                obs = get_obs(obs_pair, method, recon_model, device)
                # cal the raw obs <- recon1(recon2(...recon{n-1}(obs)...))
                if recon_chain:
                    obs = recon_chain(obs)
                actions = model(th.Tensor(obs).to(device))
            match method:
                case "recon":
                    method_loss.append(mse_loss(th.Tensor(obs_pair[0]), obs.cpu()))
                case "new":
                    if ref_model:
                        old_obs = get_obs(obs_pair, "old", device)
                        ref_actions = ref_model(th.Tensor(old_obs).to(device))
                        method_loss.append(mse_loss(ref_actions, actions).item())
            next_obs, rew, done, infos = envs.step(actions.cpu().numpy())
            rew_sum += rew
            obs_pair = next_obs
            if done:
                episodic_returns.append(rew_sum)
                break

    return episodic_returns, method_loss


def get_buf(
    args,
    train_env: gym.Env,
    old_policy: nn.Module,
    size: int,
    device: th.device,
    buf: Union[RawReplayBuffer, VanillaOverlapBuffer],
    recon_chain=None,
    recon_base_old: bool = True,
):
    more_info = isinstance(buf, VanillaOverlapBuffer)
    obs_pair = train_env.reset()
    if hasattr(old_policy, "reset"):
        old_policy.reset()
    rew_sum = 0
    rew_mean = []
    cur_step = 0
    for t in range(size):
        cur_step += 1
        old_obs, new_obs = dco(obs_pair)
        with th.no_grad():
            ready_obs = th.Tensor(old_obs).to(device)
            if recon_chain is not None:
                if not recon_base_old:
                    ready_obs = th.Tensor(new_obs).to(device)
                ready_obs = recon_chain(ready_obs)
            actions = old_policy(ready_obs).cpu().numpy()
        next_obs_pair, rew, done, infos = train_env.step(actions)
        rew_sum += rew

        real_done = done
        if done and cur_step == 1000:
            real_done = ~real_done

        if more_info:
            next_obs_pair_tmp = dco(next_obs_pair)
            buf.add(old_obs, new_obs, actions, rew, next_obs_pair_tmp[0], next_obs_pair_tmp[1], real_done)
        else:
            buf.add(old_obs, actions, rew, new_obs, real_done)

        if done:
            # print(f"[{t}/{args.total_steps}][training] episodes_retun: {rew_sum}")
            rew_mean += [rew_sum]
            rew_sum = 0
            cur_step = 0
            obs_pair = train_env.reset()
            if hasattr(old_policy, "reset"):
                old_policy.reset()
        else:
            obs_pair = next_obs_pair
    return buf, np.mean(rew_mean)


def get_noised_obs(obs, act_dim, eps, device: th.device):
    M, N, A = obs.shape[0], obs.shape[1], act_dim
    size = 20
    delta_s = 2 * eps * (th.rand(size, N, device=device) - 0.5)
    tmp_obs = obs.reshape(-1, 1, N).repeat(1, size, 1).reshape(-1, N)
    delta_s = delta_s.reshape(1, size, N).repeat(M, 1, 1).reshape(-1, N)
    noised_obs = tmp_obs + delta_s
    return M, A, size, noised_obs, delta_s


def cal_policy_smooth_loss(
    policy: nn.Module,
    policy_mean: th.Tensor,
    obs: th.Tensor,
    act_dim: int,
    policy_smooth_eps: float,
    policy_smooth_reg: float,
    device: th.device,
):
    M, A, size, noised_obs, _ = get_noised_obs(obs, act_dim, policy_smooth_eps, device)
    noised_policy_mean = policy(noised_obs)

    kl_loss = mse_loss(policy_mean.reshape(-1, 1, A).repeat(1, size, 1).reshape(-1, A), noised_policy_mean, reduce=False)
    kl_loss = kl_loss.sum(axis=-1).reshape(M, size)
    max_id = th.argmax(kl_loss, axis=1)
    kl_loss_max = kl_loss[np.arange(M), max_id].mean()
    policy_loss = policy_smooth_reg * kl_loss_max

    return policy_loss
