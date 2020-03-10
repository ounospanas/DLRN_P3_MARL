# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
import torch.nn.functional as F
from utilities import soft_update, transpose_to_tensor, transpose_list

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class MADDPG:
    def __init__(self, discount_factor=0.99, tau= 1e-2):
        super(MADDPG, self).__init__()

        #in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic, hidden_out_critic
        # critic input = obs_full = 48
        
        self.maddpg_agent = [DDPGAgent(24, 256, 128, 2, 48, 256, 128), 
                             DDPGAgent(24, 256, 128, 2, 48, 256, 128)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors
    
    def reset(self):
        for ddpg_agent in self.maddpg_agent:
            ddpg_agent.reset()

    def act(self, obs_all_agents, noise=0.0):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(obs, noise) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def update(self, samples, agent_number):
        """update the critics and actors of all the agents """

        # need to transpose each element of the samples
        # to flip obs[parallel_agent][agent_number] to
        # obs[agent_number][parallel_agent]
        obs, obs_full, action, reward, next_obs, next_obs_full, done = map(transpose_to_tensor, samples)
        obs_full = torch.stack(obs_full)
        next_obs_full = torch.stack(next_obs_full)
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)
        
        with torch.no_grad():
            q_next = agent.target_critic(next_obs_full.t(),target_actions)
        
        y = reward[agent_number].view(-1, 1) + self.discount_factor * q_next * (1 - done[agent_number].view(-1, 1))
        action = torch.cat(action, dim=1)

        q = agent.critic(obs_full.t(), action)
        

        critic_loss = F.mse_loss(q, y.detach())
        critic_loss.backward()

        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]
 
        q_input = torch.cat(q_input, dim=1)
    
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        # get the policy gradient
        actor_loss = -agent.critic(obs_full.t(), q_input).mean()
        actor_loss.backward()
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
       
    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)