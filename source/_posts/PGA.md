---
title: RL算法之策略梯度算法（Policy Gradient Algorithm)
date: 2024-09-23 17:45:59
tags: 
- Reinforcement Learning
categories:
- Reinforcement Learning
keywords:
- Reinforcement Learning
cover: https://pic.imgdb.cn/item/66f14bf3f21886ccc0bfacf5.jpg
description: 记录RL中的重要算法策略梯度算法
---
# 策略梯度算法的基本概念

策略梯度算法（Policy Gradient Algorithm）是一种直接优化策略的方法。与基于值函数的方法（如DQN）不同，策略梯度算法直接对策略进行参数化，并通过优化一个目标函数（通常是期望回报）来学习最优策略。

策略梯度算法的基本思想是：通过优化策略的参数，使得在与环境交互时，智能体能够获得最大化的累积回报。

## 优化目标

策略梯度算法是为了优化这么一个目标函数：

$$
J(\theta)=\mathbb{E}\left[\sum_{t=0}^\infty\gamma^tR_{t+1}\right]
$$

其中：

- $π_{\theta}$表示当前参数化的策略
- $γ$是折扣因子
- $R_{t+1}$是智能体在时间t时执行动作得到的reward

且可以证明，$J(\theta)=\overline{V_{\pi}}$

## REINFORECE算法

先对目标函数求梯度

$$
\nabla_\theta J(\theta)=\nabla_\theta\mathbb{E}\left[\sum_{t=0}^\infty\gamma^tR_{t+1}\right]
$$

可以写成如下的形式

$$
\nabla_\theta J(\theta)=\sum_{s\in\mathcal{S}}\eta(s)\sum_{a\in\mathcal{A}}\nabla_\theta\pi(a|s,\theta)q_\pi(s,a)=\mathbb{E}\big[\nabla_\theta\ln\pi(A|S,\theta)q_\pi(S,A)\big]
$$

我们要知道这个式子的expectation，而在实际情况中，我们是无法得到的。但根据stochastic gradient descent的基本思路，我们可以用采样来近似这个expectation，得到

$$
\nabla_{\theta}J(\theta){\approx}\nabla_{\theta}\ln\pi(a|s,\theta)q_{\pi}(s,a)
$$

可以把$q_\pi$替换成$q_t$

$$
\theta_{t+1}=\theta_t+\alpha\nabla_\theta\ln\pi(a_t|s_t,\theta_t)q_t(s_t,a_t)
$$

采取Monte-Carlo方法

$$
q_t(s_t, a_t)=\sum_{k=t+1}^{T}\gamma^{k-t-1}r_k
$$

得到的这样一个算法便称为REINFORCE。

### 伪代码

REINFORCE算法的伪代码

**Pseudocode: Policy Gradient by Monte Carlo (REINFORCE)**

**Initialization:** A parameterized function $\pi(a|s,\theta)$, $\gamma\in(0, 1)$, and $\alpha>0$.

**Aim:** Search for an optimal policy maximizing $J(\theta)$.

For the $k^{th}$ iteration, do

&nbsp;&nbsp;&nbsp;&nbsp;$s_0$ and generate an episode following $\pi(\theta_k)$.Suppose the episode is $\{s_0,a_0,r_1,\ldots,s_{T-1},a_{T-1},r_T\}$.

&nbsp;&nbsp;&nbsp;&nbsp;For $(t = 0, 1,\ldots, T-1)$, do

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Value update:**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$q_t(s_t,a_t)=\sum_{k=t+1}^T\gamma^{k-t-1}r_k$

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Policy update:**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\theta_{t+1}=\theta_t+\alpha\nabla_\theta\ln\pi(a_t|s_t,\theta_t)q_t(s_t,a_t)$

&nbsp;&nbsp;&nbsp;&nbsp;$\theta_k = \theta_T$

### 代码实现

#### 定义PolicyNet

```py
class PolicyNet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(PolicyNet,self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # softmax()函数实现数据在(0,1)上的归一化
        return F.softmax(self.fc2(x), dim=1)
```

#### 定义REINFORCE算法

```py
class REINFORCE:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,device):
        self.policy_net = R_Net.PolicyNet(state_dim,hidden_dim,action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),lr=learning_rate)
        self.gamma = gamma
        self.device = device

    def take_action(self,state):
        state = torch.tensor([state],dtype=torch.float).to(self.device)
        probs = self.policy_net(state)
        # 创建一个类别分布
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
  
    def update(self,transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        # 反向遍历
        for i in reversed(range(len(reward_list))):  
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))
            # 最后一个时间步开始反向遍历奖励列表，这样可以逐步累积reward
            G = self.gamma * G + reward
            # 每一步的损失函数
            loss = -log_prob * G
            loss.backward() 
        # 对每个参数做梯度下降
        self.optimizer.step()
```

#### 开始训练

```py
learning_rate = 1e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.98

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
env.reset(seed=0)
torch.manual_seed(0)
# 获取状态
state_dim = env.observation_space.shape[0]
# 获取动作空间的维度
action_dim = env.action_space.n

agent = R_Algorithm.REINFORCE(state_dim, hidden_dim, action_dim, learning_rate, gamma,
                  device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': []
            }
            state = env.reset()
            state = state[0]
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            return_list.append(episode_return)
            agent.update(transition_dict)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('REINFORCE on {}'.format(env_name))
plt.show()
```

运行代码，得到策略总回报Return与训练次数Episode的关系图

{% image https://pic.imgdb.cn/item/66f2605df21886ccc099c689.png, width=400px %}

可以看到，随着收集到的轨迹越来越多，REINFORCE算法有效地学习到了最优策略。不过，相比于前面的DQN算法，REINFORCE算法使用了更多的序列，这是因为REINFORCE算法是一个在线策略算法，之前收集到的轨迹数据不会被再次利用。此外，REINFORCE算法的性能也有一定程度的波动，这主要是因为每条采样轨迹的回报值波动比较大，这也是REINFORCE算法主要的不足。
