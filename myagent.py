import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def action_to_index(action):
    fr, fc, tr, tc = action
    return ((fr * 6 + fc) * 6 + tr) * 6 + tc

def index_to_action(idx):
    tc = idx % 6
    idx //= 6
    tr = idx % 6
    idx //= 6
    fc = idx % 6
    idx //= 6
    fr = idx
    return (fr, fc, tr, tc)

class ActorCriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(36, 100),
            nn.ReLU(),
        )
        self.actor = nn.Linear(100, 1296)   
        self.critic = nn.Linear(100, 1)     

    def forward(self, s):
        x = self.feature_extractor(s)
        h = self.actor(x)
        v = self.critic(x)      #Action Value function
        return h, v

class ACAgent:
    def __init__(self, entropy_coef=0.01):
        self.device = "cpu"
        self.gamma = 0.99
        self.entropy_coef = entropy_coef
        self.net = ActorCriticNet().to(self.device)

        self.actor_optimizer = optim.Adam(list(self.net.feature_extractor.parameters()) + list(self.net.actor.parameters()),lr=0.0001,)
        self.critic_optimizer = optim.Adam(list(self.net.feature_extractor.parameters()) + list(self.net.critic.parameters()),lr=0.001,)

    def preprocess_obs(self, obs, agent_name):
        board = np.array(obs, dtype=np.float32)
        if agent_name == "player_2":
            board = -board
        return torch.tensor(board.flatten(), dtype=torch.float32, device=self.device)

    def legal_action_mask(self, legal_moves):
        mask = torch.zeros(1296, dtype=torch.bool, device=self.device)
        for move in legal_moves:
            mask[action_to_index(move)] = True
        return mask

    def select_action(self, obs, legal_moves, agent_name):
        s = self.preprocess_obs(obs, agent_name).unsqueeze(0)
        h, v = self.net(s)
        h = h.squeeze(0)
        v = v.squeeze(0)
        mask = self.legal_action_mask(legal_moves)  #filling all the illegal moves with -10^^9
        masked_h = h.masked_fill(~mask, -1e9)

        pi = Categorical(logits=masked_h)     #Softmax policy defined here
        action_idx = pi.sample()
        a = index_to_action(action_idx.item())
        log_pi = pi.log_prob(action_idx)   # log pi(a|s,theta)
        entropy = pi.entropy()

        return a, log_pi, v, entropy

    def get_value(self, obs, agent_name):
        s = self.preprocess_obs(obs, agent_name).unsqueeze(0)
        _, v = self.net(s)
        return v.squeeze(0)

    def update(self, log_pi, v, reward, v_next, done, entropy):
        r = torch.tensor([reward], dtype=torch.float32, device=self.device)

        # TD target: R + gamma * v_hat(S', w)
        if done:
            td_target = r
        else:
            td_target = r + self.gamma * v_next.detach()

        # delta = R + gamma * v_hat(S', w) - v_hat(S, w)
        delta = td_target - v

        # ---- Critic update ----
        critic_loss = delta.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        # ---- Actor update ----
        actor_loss = -(log_pi * delta.detach()) - self.entropy_coef * entropy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def save(self, path="ac_checkers.pt"):
        torch.save(self.net.state_dict(), path)

    def load(self, path="ac_checkers.pt"):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.eval()