import torch as th
import torch
from torch import nn
from torch.distributions.normal import Normal
import gym
import numpy as np
from tqdm import tqdm
import pickle
from typing import Optional, Union, Callable, Final

# from typing_extensions import Final, Optional, Union, Callable
from gym.spaces import Tuple as GymTuple


DEBLOG: Final = False


class ZFilterClipWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, stat_bin_name: str):
        super().__init__(env)
        self.observation_space = GymTuple((env.observation_space, env.observation_space))
        run_stat = pickle.load(open(f"envs/{stat_bin_name}", "rb"))
        self.mean = run_stat["mean"]
        self.std = run_stat["std"]
        if DEBLOG:
            print("\033[91m" + "self.mean: " + "\033[92m", self.mean)
            print("\033[91m" + "self.std: " + "\033[92m", self.std)

    def observation(self, obs: np.array) -> GymTuple:
        wocar_obs = np.clip((obs - self.mean) / (self.std + 1e-8), -10.0, 10.0)
        overlap_obs = obs
        return (wocar_obs, overlap_obs)


def make_env(env_id: str, idx, capture_video: bool, run_name: str, stat_bin_name: str) -> Callable:
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        env = ZFilterClipWrapper(env, stat_bin_name)
        return env

    return thunk


class EnvInfo:
    def __init__(self, env_id: str, use_wocar: bool = False) -> None:
        run_stat = pickle.load(open(f"envs/{env_id} {'wocar' if use_wocar else 'natural'}.bin", "rb"))
        self.env_id = env_id
        self.mean = run_stat["mean"]
        self.std = run_stat["std"]
        tmp_env = gym.make(env_id)
        self.obs_space = tmp_env.observation_space
        self.act_space = tmp_env.action_space
        self.obs_dim = np.prod(tmp_env.observation_space.shape)
        self.act_dim = np.prod(tmp_env.action_space.shape)
        tmp_env.close()


class WocarPolicyModel(nn.Module):
    def __init__(self, env_info: EnvInfo):
        super().__init__()
        self.activation = nn.Tanh()
        self.action_dim = env_info.act_dim
        prev_size = env_info.obs_dim
        self.affine_layers = nn.ModuleList()
        hidden_sizes = (64, 64)
        for i in hidden_sizes:
            self.affine_layers.append(nn.Linear(prev_size, i, bias=True))
            prev_size = i
        self.final_mean = nn.Linear(prev_size, self.action_dim, bias=True)
        stdev_init = th.zeros(self.action_dim)
        self.log_stdev = th.nn.Parameter(stdev_init)

    def forward(self, x: th.Tensor) -> th.Tensor:
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        means = self.final_mean(x)
        # during test or inference stage, just use mean
        # std = th.exp(self.log_stdev.expand_as(means))
        # probs = Normal(means, std)
        # action = probs.sample()
        return means

    def reset(self):
        pass


class WocarLstmPolicyModel(nn.Module):
    def __init__(self, env_info: EnvInfo):
        super().__init__()
        self.hidden_sizes = (64, 64)
        self.action_dim = env_info.act_dim
        self.embedding_layer = nn.Linear(env_info.obs_dim, self.hidden_sizes[0])
        self.lstm = nn.LSTM(input_size=self.hidden_sizes[0], hidden_size=self.hidden_sizes[1], num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(self.hidden_sizes[-1], self.action_dim)
        stdev_init = th.zeros(self.action_dim)
        self.log_stdev = th.nn.Parameter(stdev_init)
        self.hidden = [th.zeros(1, 1, self.hidden_sizes[1]), th.zeros(1, 1, self.hidden_sizes[1])]

    def forward(self, x):
        assert isinstance(x, th.Tensor) and x.ndim == 2
        assert x.size(0) == 1
        embedding = self.embedding_layer(x).unsqueeze(0)
        _, hidden = self.lstm(embedding, self.hidden)
        output = self.output_layer(hidden[0])
        self.hidden[0] = hidden[0]
        self.hidden[1] = hidden[1]
        means = output.squeeze(0)
        # std = th.exp(self.log_stdev)
        return means

    def reset(self):
        self.hidden = [th.zeros(1, 1, self.hidden_sizes[1]), th.zeros(1, 1, self.hidden_sizes[1])]


def evaluate(
    model: nn.Module,
    envs: gym.Env,
    eval_episodes: int,
    device: torch.device = torch.device("cpu"),
) -> list:
    model.eval()

    episodic_returns = []
    for i in tqdm(range(eval_episodes)):
        model.reset()
        old_obs, new_obs = envs.reset()
        rew_sum = 0
        while True:
            with th.no_grad():
                actions = model(torch.Tensor(old_obs).to(device))
            next_obs, rew, done, infos = envs.step(actions.cpu().numpy())
            rew_sum += rew
            old_obs, new_obs = next_obs
            if done:
                episodic_returns.append(rew_sum)
                break

    return episodic_returns


def main(use_wocar: bool, test_env_idx: int, eval_episodes: int) -> None:
    env_id = ["Ant-v2", "HalfCheetah-v2", "Hopper-v2", "Walker2d-v2"][test_env_idx]
    env_rep = ["ant", "halfcheetah", "hopper", "walker"][test_env_idx]
    env_info = EnvInfo(env_id, use_wocar)

    ckpt_path = f"models/{env_rep}_{'wocar' if use_wocar else 'natural'}.pt"
    ckpt = th.load(ckpt_path)
    model = WocarLstmPolicyModel(env_info) if test_env_idx == 0 and use_wocar else WocarPolicyModel(env_info)
    model.load_state_dict(ckpt, strict=False)

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                env_id,
                idx=i,
                capture_video=False,
                run_name="test",
                stat_bin_name=f"{env_id} {'wocar' if use_wocar else 'natural'}.bin",
            )
            for i in range(1)
        ]
    )

    res = evaluate(model, envs, eval_episodes=eval_episodes)
    print("\033[91m" + f'Summary for : {env_id} {"wocar" if use_wocar else "natural"}' + "\033[92m")
    print(
        "\033[91m"
        + "return "
        + "\033[92m"
        + "mean: {:.2f}, std: {:.2f}, min: {:.2f}, max: {:.2f}".format(np.mean(res), np.std(res), np.min(res), np.max(res))
    )
    print("\033[91m" + "detail " + "\033[92m", np.array(res).squeeze())


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    # test single model
    main(True, 3, 10)
    main(False, 3, 10)

    # test all 8 models
    # for _env in range(4):
    #     for _wocar in (False, True):
    #         main(_wocar, _env, 10)
