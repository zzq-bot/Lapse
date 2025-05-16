import torch as th
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import mse_loss
from torch.nn import BCELoss, MSELoss, L1Loss
from models import *
from utils import EnvInfo, cal_policy_smooth_loss
from copy import deepcopy as dco
from abc import ABC, abstractmethod
import itertools


class Cirno(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def policy(self):
        pass

    @abstractmethod
    def recon(self):
        pass


class CGANAgent(Cirno):
    def __init__(self, args, env_info: EnvInfo, device):
        self.G = MLP(env_info.obs_dim, args.hidden_sizes, env_info.obs_dim).to(device)
        self.D = Discriminator(env_info.obs_dim * 2, args.hidden_sizes).to(device)
        self.G_optimizer = make_optim(args, self.G.parameters())
        self.D_optimizer = make_optim(args, self.D.parameters())
        self.bce_Loss = BCELoss().cuda()
        self.device = device
        self.args = args

    def train(self, train_obs, train_y):
        # recon train
        self.D.zero_grad()
        D_result = self.D(train_obs, train_y).squeeze()
        D_real_loss = self.bce_Loss(D_result, th.ones(D_result.size(), device=self.device))

        G_result = self.G(train_obs)
        D_result = self.D(train_obs, G_result).squeeze()
        D_fake_loss = self.bce_Loss(D_result, th.zeros(D_result.size(), device=self.device))

        D_train_loss = (D_real_loss + D_fake_loss) * 0.5
        D_train_loss.backward()
        self.D_optimizer.step()

        # train generator G
        self.G.zero_grad()

        G_result = self.G(train_obs)
        D_result = self.D(train_obs, G_result).squeeze()

        G_train_loss = self.bce_Loss(
            D_result, th.ones(D_result.size(), device=self.device)
        ) + self.args.cgan_l2_lambda * mse_loss(G_result, train_y)
        G_train_loss.backward()
        self.G_optimizer.step()

        return G_result, {"train/recon_loss": G_train_loss.item(), "train/D_loss": D_train_loss.item()}

    @property
    def policy(self):
        pass

    @property
    def recon(self):
        return self.G


class TD3BCAgent:
    def __init__(self, args, env_info: EnvInfo, noise_ball, device):
        self.actor = TD3BC_Actor(env_info).to(device)
        self.actor_target = dco(self.actor)
        self.actor_optimizer = th.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic = TD3BC_Critic(env_info).to(device)
        self.critic_target = dco(self.critic)
        self.critic_optimizer = th.optim.Adam(self.critic.parameters(), lr=args.lr)
        self.device = device
        self.noise_ball = noise_ball
        self.env_info = env_info
        self.args = args

    def train(
        self,
        train_obs,
        train_a,
        train_rew,
        train_next_obs,
        train_done,
        adapt_alpha,
        policy_update: bool,
        bc_coef: float = 1.0,
    ):
        # td3bc train
        with th.no_grad():
            noise = (th.randn_like(train_a, device=self.device) * self.args.policy_noise).clamp(
                -self.args.noise_clip, self.args.noise_clip
            )
            next_action = (self.actor_target(train_next_obs) + noise).clamp(
                self.env_info.act_space.low[0], self.env_info.act_space.high[0]
            )

            target_Q1, target_Q2 = self.critic_target(train_next_obs, next_action)
            target_Q = th.min(target_Q1, target_Q2)
            target_Q = train_rew + self.args.gamma * target_Q * (1 - train_done)

        current_Q1, current_Q2 = self.critic(train_obs, train_a)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if policy_update:
            pi = self.actor(train_obs)

            policy_smooth_loss = 0
            if self.args.policy_smooth:
                policy_smooth_loss = cal_policy_smooth_loss(
                    self.actor, pi, train_obs, self.env_info.act_dim, self.noise_ball, self.args.smooth_reg, self.device
                )

            Q = self.critic.Q1(train_obs, pi)
            lmbda = adapt_alpha / Q.abs().mean().detach()
            bc_loss = F.mse_loss(pi, train_a)
            actor_loss = -lmbda * Q.mean() + bc_coef * bc_loss + policy_smooth_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

            q_val_log = Q.mean()

            ret_log = {
                "train/bc_loss": bc_loss.item(),
                "train/qf_loss": critic_loss.item(),
                "train/q_val_mean": q_val_log.item(),
            }
            if self.args.policy_smooth:
                ret_log["train/policy_smooth_loss"] = policy_smooth_loss.item()
            return ret_log
        return {}

    @property
    def policy(self):
        return self.actor

    @property
    def recon(self):
        pass
