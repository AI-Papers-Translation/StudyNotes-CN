# 基本概念

## 机器学习

- 有监督学习（输入数据 + 目标数据）
  - 回归问题（连续输入 + 连续目标）
  - 分类问题（连续输入 + 离散目标）
  - 分类问题（离散输入 + 离散目标）
  - 特征问题
- 无监督学习（输入数据）
  - 聚类问题（连续输入 + 离散输出）
  - 降维问题（高维输入 + 低维输出）

## 线性代数

- 线性方程组
  - 欠定问题：数据少于参数个数，无数解
  - 正定问题：数据等于参数个数，可能 惟一解
  - 超定问题：数据多于参数个数，可能 没有解
    - 需要增加限制条件，转化为最优化问题

## 概率统计

- 概率[^Morris] P235
  - 似然函数：将观测数据的联合概率密度函数或者联合概率分布函数表示为参数的函数。例如：观测数据 $X,\mathbf{y}$ 已知，参数 $\mathbf{w}$ 未知，函数为 $p(\mathbf{y}|X,\mathbf{w})$ 
  - 先验分布：在得到观测数据之前参数的分布，例如： $p(\mathbf{w}|\alpha,\beta)$ 
  - 后验分布：在给定观测数据之后参数的分布，例如： $p(\mathbf{w}|X,\mathbf{y})\propto p(\mathbf{y}|X,\mathbf{w})p(\mathbf{w}|\alpha,\beta)$ 
- 估计
  - 最大似然估计：不基于先验分布和损失函数来估计参数，而使似然函数最大来估计参数。
  - 最大后验估计：

## 符号表

- 向量： $\mathbf{x}=(x_1,x_2,\cdots,x_D)^T$ 
  - $D$维特征 $\mathbf{x}_n$
  - 增广向量 $\dot{\mathbf{x}}_n = (1,\mathbf{x}_n^T)^T$
- 实数集： $\mathcal{R}$ 
- 高斯分布： $\mathcal{N}(\mu,\sigma^2)$ 
  - 均值为 μ，方差为 σ^2^ 
- 梯度算子： $\nabla_{\mathbf{w}} J(\mathbf{w}) = [\frac\partial{\partial w_0},\frac\partial{\partial w_1},\cdots,\frac\partial{\partial w_D}]^T$ 