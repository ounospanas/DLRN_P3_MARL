[//]: # (Image References)

[image1]: https://raw.github.com/ounospanas/DLRN_P3_MARL/master/maddpg_score.png "scores"
[image2]:  https://raw.github.com/ounospanas/DLRN_P3_MARL/master/maddpg.png "architecture"

### Learning Algorithm
This implementation is based on a DDPG (Deep Deterministic Policy Gradient) model, which is a Model-Free Off-policy RL algorithm applied mostly to continuous action spaces. The DDPG agent has 2 models called actor (for selecting actions) and critic (for judging how good are the actions taken by the actor), and its goal is to make both models perfom better over time by exploring and exploiting the environment. For the MARL version the implemetation introduced in [Multi Agent Actor Critic for Mixed Cooperative Competitive environments](https://papers.nips.cc/paper/7217-multi-agent-actor-critic-for-mixed-cooperative-competitive-environments.pdf) paper by OpenAI was developed. The actors of the two tennis agents have different observations, but their critics have the same observations as input (i.e., the individual observations are concatenated in one vector). The figure below is from the above paper and presents the overal architecture.

 ![architecture][image2]
#### Hyperparameters
A deep fully connected neural network was used for both critic and actor, having 2 hidden layers. Their first layer contains 256 neurons, while the second 128. 
Some other hyperparameters are:
- replay buffer size: int(1e5)
- batch size: 512
- gamma: 0.99
- tau (soft update target parameter): 1e-2  
- learning rate for critic: 3e-4 
- learning rate for actor: 3e-4 
- weight decay: 0.0
- starting noise: 1
- noise reduction: 0.99999
- noise mu: 0.0
- noise theta: 0.15
- noise sigma: 0.2

### Plot of Rewards

![scores][image1]
As shown in the figure above, the environment was solved after 9890 episodes.

### Ideas for Future Work
- as the scores plot displays, the algorithm receives a lot of null rewards for many episodes (around 2000) starting to learn. Probably a prioritized replay could select more samples that the agent achieved to get a non-zero reward
- use another policy-based algorithm for continuous control such as PPO or another actor-critic implementation such as A3C, A2C, IMPALA etc
- use another MARL algorithm such as [Mean Field Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1802.05438)
- add noise directly to the agent's parameters, which can lead to more consistent exploration and a richer set of behaviors [OpenAI paper](https://arxiv.org/abs/1706.01905)
- continue finetuning the hyperparameters to achieve a faster solution.
