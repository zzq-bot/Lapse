import torch as th
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
import random
from utils import EnvInfo


class ModelChain:
    def __init__(self) -> None:
        self.model_chain = []

    def __call__(self, X: th.Tensor) -> th.Tensor:
        for model in self.model_chain[::-1]:
            X = model(X)
        return X

    def append(self, model: nn.Module):
        self.model_chain.append(model)


class LapsePolicys:
    def __init__(self, overlap_times: int, wocar_model: nn.Module, pruning: bool = False, pruning_val: float = 0.2):
        self.overlap_times = overlap_times
        self.recon_models = [-1]
        self.off_models = [wocar_model]
        self.kappas = [0.0]
        self.wocar_model = wocar_model
        self.cur_stage = 0
        self.pruning = pruning
        self.pruning_val = pruning_val
        self.pruning_stage = 0
        self.pruning_off = [0]
        self.pruning_off_cnt = 0

    def __call__(self, X):
        if self.cur_stage == 0:
            return self.wocar_model(X)

        obss = [X]
        for i in range(self.cur_stage - self.pruning_stage):
            obss.append(self.recon_models[self.cur_stage - i](obss[-1]))

        if not self.pruning:
            last_pi = self.wocar_model(obss[-1])
            for i in range(1, self.cur_stage + 1):
                last_pi = self.kappas[i] * last_pi + (1 - self.kappas[i]) * self.off_models[i](obss[self.cur_stage - i])
        else:
            last_pi = 0.0
            for i in range(self.pruning_stage, self.cur_stage + 1):
                if self.pruning_off[i] == 0:
                    continue
                last_pi += self.pruning_off[i] * self.off_models[i](obss[self.cur_stage - i])
        return last_pi

    def update(self, kappa: float, recon_model, off_model):
        self.kappas.append(kappa)
        self.recon_models.append(recon_model)
        self.off_models.append(off_model)
        self.cur_stage += 1
        if self.pruning:
            self.pruning_off_cnt = 0
            self.pruning_stage = 100
            self.pruning_off = [0.0 for _ in range(self.cur_stage + 1)]
            for i in range(self.cur_stage + 1):
                off_coef = (1 - self.kappas[i]) * (np.prod([self.kappas[i + 1 :]]) if i < self.cur_stage else 1.0)
                print(f"[policy ensemble]: stage_{i}, off_coef={off_coef}")
                if off_coef >= self.pruning_val:
                    self.pruning_off[i] = off_coef
                    self.pruning_stage = min(self.pruning_stage, i)
                    self.pruning_off_cnt += 1

            # normalization the coef
            off_sum = np.sum(self.pruning_off)
            for i, val in enumerate(self.pruning_off):
                self.pruning_off[i] = val / off_sum
            print(f"[policy ensemble]: {self.pruning_off}")

    def reset(self):
        if hasattr(self.wocar_model, "reset"):
            self.wocar_model.reset()

    def eval(self):
        pass


def make_optim(args, model_para):
    """get a optimizer for the input model w.r.t. args

    Returns:
        torch.optim
    """
    match args.optim.lower():
        case "adam":
            recon_optimizer = optim.Adam(
                model_para,
                lr=args.lr,
            )
        case "adamw":
            recon_optimizer = optim.AdamW(
                model_para,
                lr=args.lr,
            )
        case "rmsprop":
            recon_optimizer = optim.RMSprop(
                model_para,
                lr=args.lr,
                alpha=args.optim_alpha,
                eps=args.optim_eps,
            )
        case _:
            assert 0, f"{args.optim} is not implemented"
    return recon_optimizer


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
        return means

    def reset(self):
        pass


class WocarLstmPolicyModel(nn.Module):
    def __init__(self, env_info: EnvInfo, device: th.device = th.device("cuda")):
        super().__init__()
        self.device = device
        self.hidden_sizes = (64, 64)
        self.action_dim = env_info.act_dim
        self.embedding_layer = nn.Linear(env_info.obs_dim, self.hidden_sizes[0])
        self.lstm = nn.LSTM(
            input_size=self.hidden_sizes[0],
            hidden_size=self.hidden_sizes[1],
            num_layers=1,
            batch_first=True,
        )
        self.output_layer = nn.Linear(self.hidden_sizes[-1], self.action_dim)
        stdev_init = th.zeros(self.action_dim)
        self.log_stdev = th.nn.Parameter(stdev_init)
        self.hidden = [
            th.zeros(1, 1, self.hidden_sizes[1], device=device),
            th.zeros(1, 1, self.hidden_sizes[1], device=device),
        ]

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
        self.hidden = [
            th.zeros(1, 1, self.hidden_sizes[1], device=self.device),
            th.zeros(1, 1, self.hidden_sizes[1], device=self.device),
        ]


def mlp_factory(sizes: list, activation: nn, output_activation=nn.Identity) -> nn.Sequential:
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, I_dim: int, hidder_sizes: list, O_dim: int, activation=nn.ReLU) -> None:
        super().__init__()
        self.model = mlp_factory([I_dim] + hidder_sizes + [O_dim], activation)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.model(x)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer


class Discriminator(nn.Module):
    def __init__(self, I_dim: int, hidder_sizes: list, activation=nn.ReLU):
        super().__init__()
        self.model = mlp_factory([I_dim] + hidder_sizes + [1], activation)

    def forward(self, input, label=None):
        if label is not None:
            x = th.cat([input, label], 1)
        else:
            x = input
        x = th.sigmoid(self.model(x))
        return x


class TD3BC_Actor(nn.Module):
    def __init__(self, env_info: EnvInfo):
        super(TD3BC_Actor, self).__init__()

        self.l1 = nn.Linear(env_info.obs_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, env_info.act_dim)

        self.max_action = env_info.act_space.high[0]

    def forward(self, state):
        a = F.tanh(self.l1(state))
        a = F.tanh(self.l2(a))
        return self.max_action * th.tanh(self.l3(a))


class TD3BC_Critic(nn.Module):
    def __init__(self, env_info: EnvInfo):
        super(TD3BC_Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(env_info.obs_dim + env_info.act_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(env_info.obs_dim + env_info.act_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = th.cat([state, action], 1)

        q1 = F.tanh(self.l1(sa))
        q1 = F.tanh(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.tanh(self.l4(sa))
        q2 = F.tanh(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = th.cat([state, action], 1)

        q1 = F.tanh(self.l1(sa))
        q1 = F.tanh(self.l2(q1))
        q1 = self.l3(q1)
        return q1


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            th.nn.init.zeros_(m.weight.data)
            if m.bias is not None:
                th.nn.init.constant_(m.bias.data, 0.3)
        elif isinstance(m, nn.Linear):
            th.nn.init.normal_(m.weight.data, 0.1)
            if m.bias is not None:
                th.nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zeros_()
