---
title: RL算法之DQN与Dueling DQN
date: 2024-09-22 20:43:46
tags: 
- Reinforcement Learning
categories:
- Reinforcement Learning
keywords:
- Reinforcement Learning
cover: https://pic.imgdb.cn/item/66f012e4f21886ccc0a82190.jpg
description: 记录RL中的重要算法DQN及其改进变种
---
# DQN

  DQN（Deep Q Network）是 Q-Learning 的神经网络形式，相比于普通的Q-Learning，它做出了如下的改进与性能优化：

- 使用两个独立的神经网络：目标Q网络与当前Q网络，通过最小化损失函数来更新当前Q网络，当更新到达一定次数后，再更新目标Q网络。
- 引入经验回放池，将智能体的信息记录下来，并存储在一个回放缓冲区中。在训练时，从回放缓冲区中随机抽取一小批数据进行训练。这使样本满足独立假设，并提高样本的效率，每一个样本可以被使用多次，十分适合神经网络的梯度学习。

## DQN网络的更新原则

简单来说，DQN的更新是为了最小化这么一个损失函数：

$$
L(\theta) = \mathbb{E}_{(s, a, r, s')} \left[ \left( r + \gamma \max_{a'} Q_{\text{target}}(s', a'; \theta_{\text{target}}) - Q(s, a; \theta) \right)^2 \right]
$$

其中：

- $\theta$ 是当前 Q 网络的参数
- $\theta_{\text{target}}$ 是目标 Q 网络的参数
- $s$ 和 $a$ 是当前状态和动作
- $r$ 是即时奖励，$γ$ 是折扣因子
- $s'$ 是下一状态，$a'$ 是下一步动作

## DQN算法实现

### 首先是定义经验回放池，用于存放样本与取出样本

```py
#经验回放池
class ReplayBuffer:
    def __init__(self,capacity):
        # collections.deque双端队列，支持从两端快速地添加和删除元素,当队列达到maxlen时移除最早的元素
        self.buffer = collections.deque(maxlen=capacity)
      
        # 将数据加入buffer
    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

        #从buffer中采样数据，数据量为batch_size
    def sample(self,batch_size): 
        # 随机采样
        transitions = random.sample(self.buffer,batch_size)
        # 解包transition，将同一维度的元素聚合在一起,如所有state放在一个state列表中
        state,action,reward,next_state,done = zip(*transitions)
        return np.array(state),action,reward,np.array(next_state),done
  
        # 检查当前buffer中的数据量
    def size(self):
        return len(self.buffer)
```

### 由于我们在Cartpole环境中实现DQN，神经网络不必复杂，只需定义一个只有一层隐藏层的神经网络

```py
#一层隐藏层的神经网络
class Qnet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        # 调用torch.nn.Module父类的构造函数
        super(Qnet,self).__init__()
        self.fc1 = torch.nn.Linear(state_dim,hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim,action_dim)
      
        # 隐藏层使用ReLU激活函数（去负为0取最大）
    def forward(self,x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

在这个神经网络中，输入层输入的是环境空间中的状态，在Cartpole环境中，状态空间是一个维数为4的向量，即为（车的位置，车的速度，杆的角速度，杆尖端的速度），输出层得到的是执行所有动作后的Q。

{% folding cyan,神经网络运算 %}

简单来说，神经网络是在执行一个矩阵运算。

我们把输入的状态矩阵设为$x$，它的形状为[batch_size, state_dim]，batch_size为取出的样本数量，state_dim是状态空间的维度。

我们将$x$输入到第一个线性层fc1中，fc1计算$x=x·W_1+b_1$，其中$W_1$的形状为[state_dim, hidden_dim]。然后对计算结果应用ReLU激活函数，将所有负值变为0,这是为了增加网络的非线性性质。之后再进入fc2，进行线性运算，得到输出。

{% endfolding %}

### 接下来定义DQN算法

```py
class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,epsilon,target_update,device):
        self.action_dim = action_dim
        # 当前网络
        self.q_net = DQN_Net.Qnet(state_dim,hidden_dim,self.action_dim).to(device)
        # target网络
        self.taget_q_net = DQN_Net.Qnet(state_dim,hidden_dim,self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),lr=learning_rate)
        # 折扣因子
        self.gamma = gamma
        # ε-Greedy策略
        self.epsilon = epsilon
        # targer网络更新频率
        self.taget_update = target_update
        # 记录更新次数
        self.count = 0
        # 设备选择
        self.device = device

        # 根据ε-Greedy策略选择动作
    def take_action(self,state):
        if np.random.random() < self.epsilon:
            # 生成一个[0,action_dim-1]的随机整数
            action = np.random.randint(self.action_dim)
        else:
            # state变为一个形状为(1, 4)的PyTorch张量，代表一个状态下包含的四种信息
            state = torch.tensor([state],dtype=torch.float).to(self.device)
            # 返回state下每个动作的q值
            action = self.q_net.forward(state).argmax().item()
        return action
      
      
        # 参数更新   
    def update(self,transition_dict):
        # 将state转换为一个形状为(1, 4)的二维张量，以便将其输入到网络中
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        # 将actions转换为二维张量
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)
        # 当前q值
        q_values = self.q_net.forward(states).gather(1, actions)
        # 下个状态的最大q值
        max_next_q_values = self.taget_q_net.forward(next_states).max(1)[0].view(-1,1)
        # TD误差目标
        q_targets = rewards + self.gamma * max_next_q_values * (1-dones)
        # 均方误差损失函数
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  
        # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        self.optimizer.zero_grad()  
         # 反向传播更新参数
        dqn_loss.backward() 
        self.optimizer.step()

        if self.count % self.taget_update == 0:
            # 更新target网络
            self.taget_q_net.load_state_dict(self.q_net.state_dict()) 
        self.count += 1
```

take_action函数利用ε-Greedy策略选择输入状态为state时下一步采取什么动作。

update函数用于更新当前Q网络与目标Q网络的参数

### 最后是参数设定与开始训练

```py
import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import DQN_Net
import DQN_Algorithm
import rl_utils
np.bool8 = np.bool

lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 初始化环境，定义环境实例
env_name = 'CartPole-v0'
env = gym.make(env_name)

random.seed(0)
np.random.seed(0)
env.reset(seed=0)
torch.manual_seed(0)

replay_buffer = DQN_Net.ReplayBuffer(buffer_size)

# 获取环境状态空间的维度
state_dim = env.observation_space.shape[0]
# 获取动作空间的维度
action_dim = env.action_space.n

agent = DQN_Algorithm.DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
            target_update, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                # 找到初始状态
                state = env.reset()
                # 由于env.reset()返回值是一个元组，其中第一个元素是包含状态的NumPy数组，第二个元素是额外的信息字典，我们需要取第一个Numpy数组
                state = state[0]
                done = False
                while not done:
                    # 在state状态根据ε-Greedy选择一个动作
                    action = agent.take_action(state)
                    next_state,reward,done,truncated,_ = env.step(action)
                    done = done or truncated
                    replay_buffer.add(state,action,reward,next_state,done)
                    state = next_state
                    episode_return += reward
                    # 当replay_buffer中的数据超过设定的值后，才开始训练
                    if replay_buffer.size() > minimal_size:
                        s,a,r,ns,d = replay_buffer.sample(batch_size)
                        #将采样的值加入transition_dict中
                        transition_dict = {
                                            'states' : s,
                                            'actions' : a,
                                            'rewards' : r,
                                            'next_states' : ns,
                                            'dones' : d
                                            }
                        agent.update(transition_dict)
                # 在一个episode完成后在return_list中添加这一段的return
                return_list.append(episode_return)
              
                # 每10个episode打印一次统计信息
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                # 每完成一个episode，进度条就会更新一步
                pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()
```

运行代码，得到策略总回报Return与训练次数Episode的关系图

{% image https://pic.imgdb.cn/item/66f0221ff21886ccc0b8305f.png, width=400px %}

可以看到，在训练了大概100次之后，策略的Return陡然上升，很快收敛到最优值200。但我们也可以看到，在 DQN 的性能得到提升后，它会持续出现一定程度的震荡，这主要是神经网络过拟合到一些局部经验数据后由$argmax$运算带来的影响。

# Dueling DQN

Dueling DQN是DQN的改进算法,它能够很好地学习到不同动作的差异性，在动作空间较大的环境下非常有效。

## Dueling DQN优化之处

我们定义$A(s,a)=Q(s,a)-V(s)$,$A(s,a)$为每个动作的优势函数。Dueling DQN将价值函数$V(s)$与优势函数$A(s,a)$分别建模，作为神经网络的两个不同分支来输出，然后求和得到Q值。将状态价值函数和优势函数分别建模的好处在于：某些情境下智能体只会关注状态的价值，而并不关心不同动作导致的差异，此时将二者分开建模能够使智能体更好地处理与动作关联较小的状态。

$$
Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a') \right)
$$

这个公式中的修正部分$\left( A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a') \right)$表示从优势函数中减去其均值，从而保证所有动作的平均优势为零。Dueling DQN能更高效学习状态价值函数。每一次更新时，函数都会被更新，这也会影响到其他动作的Q值。而传统的DQN只会更新某个动作的Q值，其他动作的Q值就不会更新。因此，Dueling DQN能够更加频繁、准确地学习状态价值函数。

### Dueling DQN代码实现

#### 神经网络部分的修改

修改为输出两个分支，再求和

```py
class VAnet(torch.nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        # 调用torch.nn.Module父类的构造函数
        super(VAnet,self).__init__()
        self.fc1 = torch.nn.Linear(state_dim,hidden_dim)
        # A网络分支
        self.fc2 = torch.nn.Linear(hidden_dim,action_dim)
        # V网络分支
        self.fc3 = torch.nn.Linear(hidden_dim,1)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        A = self.fc2(x)
        V = self.fc3(x)
        # A.mean(1)对A在动作维度求平均，变为一维，需要view(-1,1)reshape为二维
        Q = V + A - A.mean(1).view(-1,1)
        return Q
```

#### 算法部分

```py
      # DQN算法，包括Double DQN和Dueling DQN
class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,epsilon,target_update,device,dqn_type = 'VanillaDQN'):
        self.action_dim = action_dim

        # Dueling DQN采取不同的网络框架
        if dqn_type == 'DuelingDQN':
           self.q_net = D_DQN_Net.VAnet(state_dim,hidden_dim,self.action_dim).to(device)
           self.target_q_net = D_DQN_Net.VAnet(state_dim,hidden_dim,self.action_dim).to(device)
        # 另一套采取DQN网络框架
        else:
            self.q_net = D_DQN_Net.Qnet(state_dim,hidden_dim,self.action_dim).to(device)
            self.q_net = D_DQN_Net.Qnet(state_dim,hidden_dim,self.action_dim).to(device)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(),lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device

    def take_action(self,state):
           # 生成一个[0,action_dim-1]的随机整数,若小于ε，则随机选取一个action
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state],dtype=torch.float).to(self.device)
            # 返回使得q值最大的动作
            # item()将张量中的单个元素转为Python标量
            action = self.q_net.forward(state).argmax().item()
        return action
  
        # 寻找最大的q值
    def max_q_value(self,state):
        state = torch.tensor([state],dtype=float).to(self.device)
        return self.q_net(state).max().item()
  
    def update(self,transition_dict):
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        # actions转换为张量后仍然是一维，需要通过view(-1,1)reshape一下成为二维
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards']).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)
        # 在动作维度，根据采取的动作的标号选取每个采样state的q
        q_values = self.q_net(states).gather(1,actions)
      
        # 判断使用的是DoubleDQN还是普通DQN
        # DoubleDQN先选取能取到最大q的action，然后用action更新目标网络的q
        # 普通DQN是直接获取最大的q更新目标网络
        if self.dqn_type == 'DoubleDQN':
            max_action = self.q_net.forward(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net.forward(next_states).gather(1, max_action)
        else:
            max_next_q_values = self.target_q_net.forward(next_states).max(1)[0].view(-1, 1)

        q_targets = rewards + self.gamma * max_next_q_values * (1-dones)
        
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1
```

#### 参数设定与开始训练

```py
lr = 2e-3
num_episodes = 1000
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)

state_dim = env.observation_space.shape[0]
# 获取动作空间的维度
action_dim = env.action_space.n

random.seed(0)
np.random.seed(0)
env.reset(seed=0)
torch.manual_seed(0)

replay_buffer = D_DQN_Net.ReplayBuffer(buffer_size)
agent = D_DQN_Algorithm.DQN(state_dim,hidden_dim,action_dim,lr,gamma,epsilon,target_update,device,'DuelingDQN')

return_list = []
# 进行10次大的迭代
for i in range(10):
    # 每次迭代中，每迭代总次数的十分之一就更新一次进度条
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        # 每次大迭代中执行 num_episodes/10 次的小循环
        for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                # 找到初始状态
                state = env.reset()
                # 由于env.reset()返回值是一个元组，其中第一个元素是包含状态的NumPy数组，第二个元素是额外的信息字典，我们需要取第一个Numpy数组
                state = state[0]
                done = False
                while not done:
                    # 在state状态根据ε-Greedy选择一个动作
                    action = agent.take_action(state)
                    next_state,reward,done,truncated,_ = env.step(action)
                    done = done or truncated
                    replay_buffer.add(state,action,reward,next_state,done)
                    state = next_state
                    episode_return += reward
                    # 当replay_buffer中的数据超过设定的值后，才开始训练
                    if replay_buffer.size() > minimal_size:
                        s,a,r,ns,d = replay_buffer.sample(batch_size)
                        #将采样的值加入transition_dict中
                        transition_dict = {
                                            'states' : s,
                                            'actions' : a,
                                            'rewards' : r,
                                            'next_states' : ns,
                                            'dones' : d
                                            }
                        agent.update(transition_dict)
                # 在一个episode完成后在return_list中添加这一段的return
                return_list.append(episode_return)
              
                # 每10个episode打印一次统计信息
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                # 每完成一个episode，进度条就会更新一步
                pbar.update(1)

episodes_list = list(range(len(return_list)))
mv_return = rl_utils.moving_average(return_list, 5)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Dueling DQN on {}'.format(env_name))
plt.show()
```

运行代码，训练完成后得到

{% image https://pic.imgdb.cn/item/66f029b9f21886ccc0c10afd.png, width=400px %}

{% tip bell %}
这里出现了一个问题，为什么这里收敛的速度比DQN收敛的慢很多，这不是说明Dueling DQN的性能很差吗？

了解后发现，对于Cartpole环境来说，它的动作空间只有2维，复杂度很低，所以在这种情况下，Dueling DQN不能体现出优势，又由于相对DQN较复杂的神经网络运算方法，导致效率比较低。如果将环境换为更复杂的情况，那么收敛速度将明显快于DQN。
{% endtip %}

{% folding cyan,完整源代码点这里 %}
{% link RL_Practice,https://github.com/Lor1keet/RL_Practice, %}
{% endfolding %}

{% radio checked, 环境的改变后的代码待补充...还没做 %}
