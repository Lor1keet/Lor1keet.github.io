---
title: 利用Attention Model与RL解决Routing problem
date: 2024-09-26 17:00:33
tags: 
- Reinforce Learing
- Attention Model
- 组合优化
categories:
- Reinforce Learing
- Attention Model
- 组合优化
keywords:
- Reinforce Learing
- Attention Model
- 组合优化
cover: https://pic.imgdb.cn/item/66f555f7f21886ccc00480a1.jpg
description: 记录Attention Model解决TSP、VRP等问题的思路
---
# 起源
看完了《ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS!》这篇论文，其中提到的Attention Model为解决Routing problem提供了很好的方法
- 文中提出的模型类似于 Transformer 的编码器-解码器结构。编码器生成输入节点的嵌入表示，解码器则逐步输出节点的顺序（即路径），这种方法与 Transformer 的编码器和解码器的交互非常相似。
- 他们的编码器使用了类似 Transformer 中的多头注意力机制，每层由一个多头注意力子层和一个前馈子层组成，这些子层通过跳跃连接和批归一化进行组合。

下面写一下本人啃了半天勉强学懂的模型思想。
# 流程
这篇文章根据 TSP 定义注意力模型。对于其他问题，模型是相同的，但需要相应地定义输入、掩码和解码器上下文，TSP并不需要这么多约束，作为演示非常合适。
将问题实例 $s$ 定义为具有 $n$ 个节点的图，其中节点 $i\in\{1,\ldots,n\}$ 由特征 $x_i$ 表示。对于 TSP，$x_i$ 是节点 $i$ 的坐标。定义 $\boldsymbol{\pi} = (\pi_1,\ldots,\pi_n)$ 作为节点的排列。

## 编码器(Encoder)
编码器的每个layer流程如图
{% image https://pic.imgdb.cn/item/66f52ce3f21886ccc0e29d47.png, height=400px,width=1000px %}

在TSP问题中，$x_i$ 的维度是2，通过线性变换将每个点的坐标变换为128维 $\mathrm{h}_i^{(0)}=W^\mathrm{x}\mathrm{x}_i+\mathrm{b}^\mathrm{x}$。
对于每个128维的 $\mathrm{h}_i^{(0)}$，使用不同的线性投影矩阵生成8个头的查询（$Q$）、键（$K$）与值（$V$）向量
$$Q_1=\mathbf{h}_iW_1^Q,\quad Q_2=\mathbf{h}_iW_2^Q,\quad\ldots,\quad Q_8=\mathbf{h}_iW_8^Q$$
$$K_1=\mathbf{h}_iW_1^K,\quad K_2=\mathbf{h}_iW_2^K,\quad\ldots,\quad K_8=\mathbf{h}_iW_8^K$$
$$V_1=\mathbf{h}_iW_1^V,\quad V_2=\mathbf{h}_iW_2^V,\quad\ldots,\quad V_8=\mathbf{h}_iW_8^V$$

每个输入向量就被分成了属于$Q$、$K$、$V$的各八个头。然后每个头的查询向量 $Q_{i}^{(s)}$ 与所有其他头的键向量 $K_{i}^{(s)}$ 计算点积后除以维度的根号并取 $softmax$ 函数，再与值向量点积得到
$$\mathrm{head}_{ij}=\mathrm{softmax}\left(\frac{Q_i^{(s)}K_j^{(s)^\top}}{\sqrt{d_k}}\right)V_j^{(s)}$$

其中 $(s=0,1,...,7)$，$(i，j=0,1...,n)$

除以维度的根号是防止得到的值过大

每个头的自注意力都计算完后，将每个输入的八个头拼起来
$$\mathrm{MHA}_i^\ell\left(\mathrm{h}_1^{(\ell-1)},\ldots,\mathrm{h}_n^{(\ell-1)}\right) = concat(head_{i1},...,head_{in})$$

再进行 skip-connection（跳跃连接）与 BN（Batch Normalization）

$$\begin{aligned}
\hat{\mathbf{h}}_{i}& =\mathrm{BN}^\ell\left(\mathbf{h}_i^{(\ell-1)}+\mathrm{MHA}_i^\ell\left(\mathbf{h}_1^{(\ell-1)},\ldots,\mathbf{h}_n^{(\ell-1)}\right)\right) \\
\mathbf{h}_i^{(\ell)}& =\mathrm{BN}^\ell\left(\hat{\mathbf{h}}_i+\mathrm{FF}^\ell(\hat{\mathbf{h}}_i)\right). 
\end{aligned}$$

这样得到了每个输入的输出，由于编码器有 $N=3$ 个layer，我们要进行3次这种运算，把上一层的输出作为下一层的输入。$\bar{\mathrm{h}}^{(N)} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{h}_{i}^{(N)}$ 作为全局图嵌入。

## 解码器(Decoder)
在时间步 $t$ 时解码器的上下文嵌入来自于编码器直到 $t$ 时刻的输出。当 $t=1$ 时由于还没选择节点，我们把 $\mathbf{v}^{l}$ 和 $\mathbf{v}^{f}$ 作为输入的占位符

$$\mathbf{h}_{(c)}^{(N)}=\begin{cases}\left[\bar{\mathbf{h}}^{(N)},\mathbf{h}_{\pi_{t-1}}^{(N)},\mathbf{h}_{\pi_1}^{(N)}\right]&t>1\\\left[\bar{\mathbf{h}}^{(N)},\mathbf{v}^{\mathrm{l}},\mathbf{v}^{\mathrm{f}}\right]&t=1.\end{cases}$$

然后由上下文嵌入 $\mathbf{h}_{(c)}^{(N)}$ 生成查询向量
$$Q_{(c)}=W^Qh_{(c)}^{(N)}$$

每个节点 $j$ 的键向量和值向量分别表示为
$$k_j=W^Kh_j^{(N)},\quad v_j=W^Vh_j^{(N)}$$

然后计算查询向量和每个节点键向量的兼容性
$$u(c,j)=\frac{q_{(c)}^\top k_j}{\sqrt{d_k}}$$

使用掩码操作，保证不能选择已经访问过的节点
$$u_{(c)j}=\begin{cases}\frac{\mathbf{q}_{(c)}^T\mathbf{k}_j}{\sqrt{d_k}}&\text{if} j\neq\pi_{t'}\quad\forall t'<t\\-\infty&\text{otherwise.}\end{cases}$$

在屏蔽前，为了计算最终的未归一化对数概率，添加一个具有单个注意力头的最终解码器层，使用 tanh 将结果裁剪在 [−C, C] (C = 10) 内
$$u_{(c)j}=\begin{cases}C\cdot\tanh\left(\frac{\mathbf{q}_{(c)}^T\mathbf{k}_j}{\sqrt{d_k}}\right)&\text{if} j\neq\pi_{t'}\quad\forall t'<t\\-\infty&\text{otherwise.}\end{cases}$$

最后由 $softmax$ 函数将 $u_{(c)j}$ 转化为选择节点的概率
$$p_i=p_{\boldsymbol{\theta}}(\pi_t=i|s,\pi_{1:t-1})=\frac{e^{\boldsymbol{u}_{(c)i}}}{\sum_je^{\boldsymbol{u}_{(c)j}}}.$$

这个概率表示节点 $i$ 被选为下一个访问节点的概率。

## 带回滚基线的 REINFORCE 算法

在Attention Model选择完路径之后，运用策略梯度算法中的 REINFORECE 来对结果进行训练。优化的目标是最小化路径代价
$$\mathcal{L}(\theta|s)=\mathbb{E}_{\pi\sim p_\theta(\pi|s)}[L(\pi)]$$

引入一个基线 $b(s)$ ，类似于 A2C 算法,基线用来减少梯度的方差。这个基线的引入不会影响梯度的期望值，但可以显著降低方差，从而加快训练收敛。

增加基线后，更新 REINFORCE 的梯度更新公式
$$\nabla\mathcal{L}(\theta|s)=\mathbb{E}_{\pi\sim p_\theta(\pi|s)}\left[(L(\pi)-b(s))\nabla\log p_\theta(\pi|s)\right]$$

### 训练步骤
#### 初始化模型参数 $θ$ 和基线策略参数 $θ_{BL}$
#### 逐步对每个回合和每个回合的每一步进行训练
- 随机采样一个实例 $s_i$
- 通过当前的策略 $θ$ 生成一段采样路径
- 通过贪心回滚策略 $θ_{BL}$ 生成贪心路径 作为基线 $\pi_{i}^{\mathrm{BL}}$
- 根据 $\nabla\mathcal{L}\leftarrow\sum_{i=1}^B\left(L(\boldsymbol{\pi}_i)-L(\boldsymbol{\pi}_i^{\mathrm{BL}})\right)\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(\boldsymbol{\pi}_i)$ 计算梯度
- 利用 Adam 优化器更新参数$θ$
#### 如果当前策略显著优于基线策略，则更新基线策略

## 实验结果
见论文

{% link Attention Learn to Solve Routing Problems!,https://arxiv.org/abs/1803.08475, %}

​
 