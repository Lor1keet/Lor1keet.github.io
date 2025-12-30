---
title: RL算法之Actor-Critic
date: 2024-09-23 20:25:14
tags: 
- Reinforcement Learning
categories:
- Reinforcement Learning
keywords:
- Reinforcement Learning
cover: https://pic.imgdb.cn/item/66f1666df21886ccc0d839b2.jpg
description: 记录RL中的重要算法Actor-Critic及其改进
---
# AC算法

在REINFORCE算法中，目标函数的梯度中有一项轨迹回报，用于指导策略的更新。REINFOCE算法用蒙特卡洛方法来估计$Q(s,a)$,而AC使用TD（时序差分）方法来估计。

AC算法的核心思想是同时使用两个部分：Actor（策略网络）和 Critic（价值网络）

- Actor要做的是与环境交互，并在Critic价值函数的指导下用策略梯度学习一个更好的策略
- Critic要做的是通过Actor与环境交互收集的数据学习一个价值函数，这个价值函数会用于判断在当前状态什么动作是好的，什么动作不是好的，进而帮助Actor进行策略更新

直接引入A2C（Advantage Actor-Critic）算法。

# A2C算法（Advantage Actor-Critic）

我们在最基本的AC算法中添加一项$b(S)$,作为baseline

$$
\begin{aligned}
\nabla_{\theta}J(\theta)& =\mathbb{E}_{S\sim\eta,A\sim\pi}\left[\nabla_{\theta}\operatorname{ln}\pi(A|S,\theta_{t})q_{\pi}(S,A)\right] \\
&=\mathbb{E}_{S\sim\eta,A\sim\pi}\Big[\nabla_{\theta}\ln\pi(A|S,\theta_{t})(q_{\pi}(S,A)-b(S))\Big]
\end{aligned}
$$

通常$b(s)=v_{\pi}(s)$。然后通过梯度上升得到

$$
\begin{aligned}
\theta_{t+1}& =\theta_t+\alpha\mathbb{E}\bigg[\nabla_\theta\ln\pi(A|S,\theta_t)[q_\pi(S,A)-v_\pi(S)]\bigg] \\
&\doteq\theta_t+\alpha\mathbb{E}\Big[\nabla_\theta\ln\pi(A|S,\theta_t)\delta_\pi(S,A)\Big]
\end{aligned}
$$

运用SGD得到

$$
\begin{aligned}
\theta_{t+1}& \begin{aligned}=\theta_t+\alpha\nabla_\theta\ln\pi(a_t|s_t,\theta_t)[q_t(s_t,a_t)-v_t(s_t)]\end{aligned} \\
&=\theta_t+\alpha\nabla_\theta\ln\pi(a_t|s_t,\theta_t)\delta_t(s_t,a_t)
\end{aligned}
$$

将$\delta_t$替换为TD error

$$
\delta_t=q_t(s_t,a_t)-v_t(s_t)\to r_{t+1}+\gamma v_t(s_{t+1})-v_t(s_t)
$$

## 伪代码

**Advantage actor-critic (A2C) or TD actor-critic**

**Aim**: Search for an optimal policy by maximizing $J(\theta)$.

At time step $t$ in each episode, do

&nbsp;&nbsp;&nbsp;&nbsp;Generate $a_t$ following $\pi(a | s_t, \theta_t)$ and then observe $r_{t+1}, s_{t+1}$.

&nbsp;&nbsp;&nbsp;&nbsp;**TD error (advantage function):**

&nbsp;&nbsp;&nbsp;&nbsp;$\delta_t = r_{t+1} + \gamma v(s_{t+1}, w_t) - v(s_t, w_t)$

&nbsp;&nbsp;&nbsp;&nbsp;**Critic (value update):**

&nbsp;&nbsp;&nbsp;&nbsp;$w_{t+1} = w_t + \alpha_w \delta_t \nabla_w v(s_t, w_t)$

&nbsp;&nbsp;&nbsp;&nbsp;**Actor (policy update):**

&nbsp;&nbsp;&nbsp;&nbsp;$\theta_{t+1} = \theta_t + \alpha_\theta \delta_t \nabla_\theta \ln \pi(a_t | s_t, \theta_t)$

### 代码实现

#### 定义策略网络与价值网络

```py
# 策略网络
class PolicyNet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(PolicyNet,self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # softmax()函数实现数据在(0,1)上的归一化
        return F.softmax(self.fc2(x), dim=1)

# 引入一个价值网络，输入是某个状态，输出是状态的value
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

#### 定义Actor-Critic算法

```py
class ActorCritic:
    def __init__(self,state_dim,hidden_dim,action_dim,actor_lr,critic_lr,gamma,device):
        # 策略网络
        self.actor = AC_Net.PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        # 价值网络
        self.critic = AC_Net.ValueNet(state_dim, hidden_dim).to(device)
        # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        # 价值网络优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=critic_lr) 
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
  
    def update(self,transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)
        # TD target
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -dones)
        # TD error
        td_delta = td_target - self.critic(states) 
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数
```

#### 开始训练

```py
actor_lr = 1e-3
critic_lr = 1e-2
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
agent = AC_Algorithm.ActorCritic(state_dim,hidden_dim,action_dim,actor_lr,critic_lr,gamma,device)

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
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()
```

运行代码，得到

{% image https://pic.imgdb.cn/item/66f673c7f21886ccc0fa6236.png, width=400px %}

根据实验结果我们可以发现，Actor-Critic 算法很快便能收敛到最优策略，并且训练过程非常稳定，抖动情况相比 REINFORCE 算法有了明显的改进，这说明价值函数的引入减小了方差。

# SAC（Soft Actor-Critic）

- 未完待续
