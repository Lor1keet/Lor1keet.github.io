---
title: RL算法之Actor-Critic
date: 2024-09-23 20:25:14
tags: 
- Reinforce Learing
categories:
- Reinforce Learing
keywords:
- Reinforce Learing
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
$$\begin{aligned}
\theta_{t+1}& =\theta_t+\alpha\mathbb{E}\bigg[\nabla_\theta\ln\pi(A|S,\theta_t)[q_\pi(S,A)-v_\pi(S)]\bigg] \\
&\doteq\theta_t+\alpha\mathbb{E}\Big[\nabla_\theta\ln\pi(A|S,\theta_t)\delta_\pi(S,A)\Big]
\end{aligned}$$

运用SGD得到
$$\begin{aligned}
\theta_{t+1}& \begin{aligned}=\theta_t+\alpha\nabla_\theta\ln\pi(a_t|s_t,\theta_t)[q_t(s_t,a_t)-v_t(s_t)]\end{aligned} \\
&=\theta_t+\alpha\nabla_\theta\ln\pi(a_t|s_t,\theta_t)\delta_t(s_t,a_t)
\end{aligned}$$

将$\delta_t$替换为TD error
$$\delta_t=q_t(s_t,a_t)-v_t(s_t)\to r_{t+1}+\gamma v_t(s_{t+1})-v_t(s_t)$$

## 伪代码

**Advantage actor-critic (A2C) or TD actor-critic**

**Aim**: Search for an optimal policy by maximizing $J(\theta)$.

At time step $t$ in each episode, do  
Generate $a_t$ following $\pi(a | s_t, \theta_t)$ and then observe $r_{t+1}, s_{t+1}$.

**TD error (advantage function):**
$
\delta_t = r_{t+1} + \gamma v(s_{t+1}, w_t) - v(s_t, w_t)
$

**Critic (value update):**
$
w_{t+1} = w_t + \alpha_w \delta_t \nabla_w v(s_t, w_t)
$

**Actor (policy update):**
$
\theta_{t+1} = \theta_t + \alpha_\theta \delta_t \nabla_\theta \ln \pi(a_t | s_t, \theta_t)
$

### 代码实现
- 过两天补

# SAC（Soft Actor-Critic）
- 还没学到
