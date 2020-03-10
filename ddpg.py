# individual network settings for each actor + critic pair
# see models.py for details

from models import Actor, Critic
from utilities import hard_update
from torch.optim import Adam
import torch
import numpy as np


# add OU noise for exploration
from OUNoise import OUNoise

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

class DDPGAgent:
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic, 
                 lr_actor=3.0e-4, lr_critic=3.0e-4):
        super(DDPGAgent, self).__init__()
        
        self.actor = Actor(in_actor, hidden_in_actor, hidden_out_actor, out_actor).to(device)
        self.critic = Critic(in_critic, hidden_in_critic, hidden_out_critic, 4).to(device)
        self.target_actor = Actor(in_actor, hidden_in_actor, hidden_out_actor, out_actor).to(device)
        self.target_critic = Critic(in_critic, hidden_in_critic, hidden_out_critic, 4).to(device)

        self.noise = OUNoise(out_actor, scale=1.0 )

        
        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=0)

    def reset(self):
        self.noise.reset()

    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        
        with torch.no_grad():
            action = self.actor(obs).cpu().data.numpy()
        self.actor.train()
        action += noise*self.noise.noise().cpu().data.numpy()
        return torch.Tensor(np.clip(action, -1, 1))

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        with torch.no_grad():
            action = self.target_actor(obs).cpu().data.numpy()
        self.target_actor.train()
        action += noise*self.noise.noise().cpu().data.numpy()
        return torch.Tensor(np.clip(action, -1, 1))