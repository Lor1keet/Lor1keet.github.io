---
title: 利用 gym 自定义迷宫环境与 DQN 训练实现
date: 2024-10-17 20:27:03
tags: 
- Reinforce Learing
categories:
- Reinforce Learing
keywords:
- Reinforce Learing
cover: https://pic.imgdb.cn/item/67110b74d29ded1a8c4f1371.jpg
description: 记录自定义迷宫环境与利用DQN训练
---
# 自定义环境
gym 库提供了很多的内置环境，比如最常见的 Cartpole 车杆模型。但很多时候内置的环境无法满足我们需要的环境，这个时候可以在 gym 提供的框架基础上自定义环境。

## 代码
这里以一个 5X5 的简单迷宫为例，从起点出发，撞墙或出界的奖励为-20，正常探索的奖励为-0.1，到达终点的奖励是50。

```py
import gym
from gym import spaces
import numpy as np
import random

class SimpleMazeEnv(gym.Env):
    """ 自定义简单迷宫环境 """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(SimpleMazeEnv, self).__init__()
        
        # 定义迷宫的大小
        self.grid_size = 5
        self.maze = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # 设置障碍物（墙壁）和目标
        self.maze[1, 2] = 1  # 墙壁
        self.maze[1, 3] = 1  # 墙壁
        self.maze[1, 4] = 1  # 墙壁
        self.maze[3, 1] = 1  # 墙壁
        self.maze[3, 3] = 1  # 墙壁
        self.goal = (4, 4)   # 目标位置
        
        # 定义动作空间（上下左右）
        self.action_space = spaces.Discrete(4)
        
        # 定义观测空间（智能体的当前位置）
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        
        # 初始化智能体的位置
        self.agent_pos = np.array([0, 0])
    
    def reset(self, seed=None, options=None):
        """ 重置环境，返回初始观测和信息字典 """
        # 处理随机种子
        if seed is not None:
            np.random.seed(seed) 
        
        self.agent_pos = np.array([0, 0])  # 重置智能体位置
        return np.array(self.agent_pos, dtype=np.int32), {}  # 返回元组 (obs, info)

    def step(self, action):
        """ 根据动作更新智能体的位置，返回 (观测, 奖励, 是否结束, 额外信息) """
        # 保存原始位置
        old_pos = self.agent_pos.copy()
        
        # 根据动作更新位置（0:上, 1:右, 2:下, 3:左）
        if action == 0:   # 上
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1: # 右
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        elif action == 2: # 下
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 3: # 左
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)

        # 检查是否撞到墙壁或出界
        if self.maze[tuple(self.agent_pos)] == 1 or np.all(old_pos == self.agent_pos):
            self.agent_pos = old_pos  # 如果无效则返回到原位置
            reward = -20  # 撞墙或出界的惩罚
        else:  # 如果有效
            reward = -0.1  # 每一步都给出小惩罚

        # 检查是否到达目标
        done = np.array_equal(self.agent_pos, self.goal)
        if done:
            reward = 50  # 到达目标的奖励

        truncated = False

        return np.array(self.agent_pos, dtype=np.int32), reward, done, truncated, {}

    
    def render(self, mode='human'):
        """ 渲染环境 """
        maze_render = self.maze.copy()
        maze_render[tuple(self.agent_pos)] = 2  # 智能体位置为2
        maze_render[self.goal] = 3  # 目标位置为3
        print(maze_render)
    
    def close(self):
        """ 关闭环境 """
        pass
```
环境中主要有五个模块
- init 主要是初始化一些参数
- reset 初始化环境
- step 利用智能体的运动学模型和动力学模型计算下一步的状态和立即回报，并判断是否达到终止状态
- render 绘图函数，可以为空，但必须存在
- close 关闭图形页面
  
## 添加自定义环境
写好自定义环境的代码后，我们要将文件添加到库中。将文件保存为 maze.py ，在 ...\Lib\site-packages\gym\envs\classic_control 目录中新建一个文件夹（我取名为myenv），将 maze.py 保存在这个文件夹中。

然后打开 ...\Lib\site-packages\gym\envs 目录下的 \__init__.py 文件，添加如下代码
```py
register(
    id="Maze-v0", # 环境id可自定义，但是一定要加上-v0 代表版本号
    entry_point="gym.envs.classic_control.myenv.maze:SimpleMazeEnv",
    max_episode_steps=200,
    reward_threshold=100.0,
)
```

在头文件中加入
```py
from gym.envs.classic_control.myenv.maze import SimpleMazeEnv
```
到此就完成了自定义环境的引入！

# 训练
参考 {% post_link DQN %}
只需把主程序中的 env_name 改为

```py
env_name = 'Maze-v0'
```

添加几行代码来显示训练后的最优路径和最大奖励

```py
+   all_path = []
+   episode_rewards = []
    return_list = []

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                agent.epsilon = max(final_epsilon, agent.epsilon * epsilon_decay)
                episode_return = 0
                path = []
                # state为一维数组
                state, _ = env.reset()  # env.reset()现在返回一个元组
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, truncated, _ = env.step(action)
                    done = done or truncated
                    replay_buffer.add(state, action, reward, next_state, done)
+                   path.append(state)
                    state = next_state
                    episode_return += reward
                    # 当buffer数据的数量超过一定值后,才进行Q网络训练
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
+               all_path.append(path)
+               episode_rewards.append(episode_return)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

+   max_reward_index = np.argmax(episode_rewards)  # 获取最大奖励的索引
+   best_path = all_path[max_reward_index]
+   coordinates = [tuple(int(item) for item in arr) for arr in best_path]
+   print("Best Path:", coordinates)
+   print("Max Reward:",max(return_list))

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
得到
{% image https://pic.imgdb.cn/item/67110ab9d29ded1a8c4e7e06.png, width=400px %}
{% image https://pic.imgdb.cn/item/67110afbd29ded1a8c4eb436.png, width=400px %}