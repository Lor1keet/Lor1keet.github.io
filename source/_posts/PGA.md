---
title: RL算法之策略梯度算法（Policy Gradient Algorithm)
date: 2024-09-23 17:45:59
tags: 
- Reinforce Learing
categories:
- Reinforce Learing
keywords:
- Reinforce Learing
cover: https://pic.imgdb.cn/item/66f14bf3f21886ccc0bfacf5.jpg
description: 记录RL中的重要算法策略梯度算法
---

# 策略梯度算法的基本概念
策略梯度算法（Policy Gradient Algorithm）是一种直接优化策略的方法。与基于值函数的方法（如DQN）不同，策略梯度算法直接对策略进行参数化，并通过优化一个目标函数（通常是期望回报）来学习最优策略。

策略梯度算法的基本思想是：通过优化策略的参数，使得在与环境交互时，智能体能够获得最大化的累积回报。

## 优化目标
策略梯度算法是为了优化这么一个目标函数：

$$J(\theta)=\mathbb{E}\left[\sum_{t=0}^\infty\gamma^tR_{t+1}\right]$$

其中：
- $π_{\theta}$​表示当前参数化的策略
- $γ$是折扣因子
- $R_{t+1}$是智能体在时间t时执行动作得到的reward

且可以证明，$J(\theta)=\overline{V_{\pi}}$

## REINFORECE算法
先对目标函数求梯度
$$\nabla_\theta J(\theta)=\nabla_\theta\mathbb{E}\left[\sum_{t=0}^\infty\gamma^tR_{t+1}\right]$$

可以写成如下的形式
$$\nabla_\theta J(\theta)=\sum_{s\in\mathcal{S}}\eta(s)\sum_{a\in\mathcal{A}}\nabla_\theta\pi(a|s,\theta)q_\pi(s,a)=\mathbb{E}\big[\nabla_\theta\ln\pi(A|S,\theta)q_\pi(S,A)\big]$$

我们要知道这个式子的expectation，而在实际情况中，我们是无法得到的。但根据stochastic gradient descent的基本思路，我们可以用采样来近似这个expectation，得到
$$\nabla_{\theta}J(\theta){\approx}\nabla_{\theta}\ln\pi(a|s,\theta)q_{\pi}(s,a)$$

可以把$q_\pi$替换成$q_t$

$$\theta_{t+1}=\theta_t+\alpha\nabla_\theta\ln\pi(a_t|s_t,\theta_t)q_t(s_t,a_t)$$

采取Monte-Carlo方法

$$q_t(s_t, a_t)=\sum_{k=t+1}^{T}\gamma^{k-t-1}r_k$$

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
- 过两天补

