import os
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import functional as F

from .base import Algorithm
from discor.network import TwinnedStateActionFunction, DeterministicPolicy
from discor.utils import disable_gradients, soft_update, update_params, assert_action

import logging
logger = logging.getLogger(__name__)


class TD3(Algorithm):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)
    Paper: https://arxiv.org/abs/1802.09477

    TD3 addresses function approximation errors in actor-critic methods by:
    1. Using twin critics to reduce overestimation bias
    2. Delayed policy updates for stability
    3. Target policy smoothing (adding noise to target actions)

    Well-suited for continuous control tasks with complex dynamics like racing.
    """

    def __init__(self, state_dim, action_dim, device, gamma=0.99, nstep=1,
                 policy_lr=0.0003, q_lr=0.0003, policy_hidden_units=[256, 256],
                 q_hidden_units=[256, 256], target_update_coef=0.005,
                 policy_noise=0.2, noise_clip=0.5, exploration_noise=0.1,
                 steer_exploration_noise_factor=1, acc_exploration_noise_factor=1, brake_exploration_noise_factor=1,
                 policy_delay=2, log_interval=10, seed=0):
        super().__init__(
            state_dim, action_dim, device, gamma, nstep, log_interval, seed)

        # Build networks
        self._policy_net = DeterministicPolicy(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=policy_hidden_units
        ).to(self._device)
        self._online_q_net = TwinnedStateActionFunction(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=q_hidden_units
        ).to(self._device)
        self._target_q_net = TwinnedStateActionFunction(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=q_hidden_units
        ).to(self._device).eval()
        self._target_policy_net = DeterministicPolicy(
            state_dim=self._state_dim,
            action_dim=self._action_dim,
            hidden_units=policy_hidden_units
        ).to(self._device).eval()

        # Copy parameters of the learning network to the target network
        self._target_q_net.load_state_dict(self._online_q_net.state_dict())
        self._target_policy_net.load_state_dict(self._policy_net.state_dict())

        # Disable gradient calculations of the target network
        disable_gradients(self._target_q_net)
        disable_gradients(self._target_policy_net)

        # Optimizers
        self._policy_optim = Adam(self._policy_net.parameters(), lr=policy_lr)
        self._q_optim = Adam(self._online_q_net.parameters(), lr=q_lr)

        # TD3 specific parameters
        self._policy_noise = policy_noise
        self._noise_clip = noise_clip
        self._policy_delay = policy_delay
        self._exploration_noise = exploration_noise
        self._steer_exploration_noise = self._exploration_noise * steer_exploration_noise_factor
        self._acc_exploration_noise = self._exploration_noise * acc_exploration_noise_factor
        self._brake_exploration_noise = self._exploration_noise * brake_exploration_noise_factor
        self._noise_factors = np.array([self._steer_exploration_noise,
                                        self._acc_exploration_noise,
                                        self._brake_exploration_noise] + [1.0] * (self._action_dim - 3))
        self._noise_factors = torch.tensor(self._noise_factors, dtype=torch.float, device=self._device)
        self._policy_update_counter = 0

        self._target_update_coef = target_update_coef

    def explore(self, state):
        state = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device)
        with torch.no_grad():
            action = self._policy_net(state)
            # Add exploration noise factors
            noise = torch.randn_like(action) * self._noise_factors
            action = torch.clamp(action + noise, -1.0, 1.0)
        action = action.cpu().numpy()[0]
        assert_action(action)
        return action, 0.0  # No entropy for deterministic policy

    def exploit(self, state):
        state = torch.tensor(
            state[None, ...].copy(), dtype=torch.float, device=self._device)
        with torch.no_grad():
            action = self._policy_net(state)
        action = action.cpu().numpy()[0]
        assert_action(action)
        return action, 0.0  # No entropy for deterministic policy

    def update_target_networks(self):
        soft_update(
            self._target_q_net, self._online_q_net, self._target_update_coef)
        soft_update(
            self._target_policy_net, self._policy_net, self._target_update_coef)

    def update_online_networks(self, batch, writer):
        self._learning_steps += 1

        # Update critic
        q_loss, mean_q1, mean_q2 = self.update_critic(batch, writer)

        # Delayed policy updates
        stats = {"q_loss": q_loss.detach().item(), "mean_q1": mean_q1, "mean_q2": mean_q2}

        if self._learning_steps % self._policy_delay == 0:
            # Update policy
            policy_loss = self.update_policy(batch, writer)
            stats["policy_loss"] = policy_loss.detach().item()

            # Update target networks
            self.update_target_networks()

        return stats

    def update_critic(self, batch, writer):
        states, actions, rewards, next_states, dones = batch

        # Calculate target Q value
        with torch.no_grad():
            # Select action from target policy with noise (target policy smoothing)
            next_actions = self._target_policy_net(next_states)
            noise = torch.randn_like(next_actions) * self._policy_noise
            noise = torch.clamp(noise, -self._noise_clip, self._noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)

            # Calculate target Q values
            next_q1, next_q2 = self._target_q_net(next_states, next_actions)
            # Take the min of the two target Q values to reduce overestimation
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + (1.0 - dones) * self._discount * next_q

        # Calculate current Q values
        curr_q1, curr_q2 = self._online_q_net(states, actions)

        # Calculate critic loss
        q1_loss = F.mse_loss(curr_q1, target_q)
        q2_loss = F.mse_loss(curr_q2, target_q)
        q_loss = q1_loss + q2_loss

        # Update critic
        update_params(self._q_optim, q_loss)

        # Log metrics
        if self._learning_steps % self._log_interval == 0:
            writer.add_scalar(
                'loss/Q', q_loss.detach().item(),
                self._learning_steps)
            writer.add_scalar(
                'stats/mean_Q1', curr_q1.detach().mean().item(),
                self._learning_steps)
            writer.add_scalar(
                'stats/mean_Q2', curr_q2.detach().mean().item(),
                self._learning_steps)

        return q_loss, curr_q1.detach().mean().item(), curr_q2.detach().mean().item()

    def update_policy(self, batch, writer):
        states, _, _, _, _ = batch

        # Calculate policy loss
        actions = self._policy_net(states)
        q1, _ = self._online_q_net(states, actions)
        policy_loss = -q1.mean()

        # Update policy
        update_params(self._policy_optim, policy_loss)

        if self._learning_steps % self._log_interval == 0:
            writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self._learning_steps)

        return policy_loss

    def save_models(self, save_dir):
        super().save_models(save_dir)
        self._policy_net.save(os.path.join(save_dir, 'policy_net.pth'))
        self._online_q_net.save(os.path.join(save_dir, 'online_q_net.pth'))
        self._target_q_net.save(os.path.join(save_dir, 'target_q_net.pth'))
        self._target_policy_net.save(os.path.join(save_dir, 'target_policy_net.pth'))

    def load_models(self, load_dir):
        self._policy_net.load(os.path.join(load_dir, 'policy_net.pth'))
        self._online_q_net.load(os.path.join(load_dir, 'online_q_net.pth'))
        self._target_q_net.load(os.path.join(load_dir, 'target_q_net.pth'))
        self._target_policy_net.load(os.path.join(load_dir, 'target_policy_net.pth'))
