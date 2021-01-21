# Lagrange 对偶性

在约束最优化问题中，使用 Language 对偶性 ( Duality ) 将原始问题转化为对偶问题，通过解对偶问题而得到原始问题的解，这个方法应用在许多统计学习方法中。例如：「最大熵模型」与「支持向量机」。

## 1. 原始问题

假设 $f ( \text{x} ) ,c_i ( \text{x} ) ,h_j ( \text{x} )$ 是定义在 $\mathbb{R}^n$ 上的连续可向函数，其中 $\text{x}= ( x_1,\cdots,x_n )^T \in\mathbb{R}^n$，需要求解约束优化问题 ( 原始问题 )
$$
\begin{aligned}
  &\min_{\text{x}\in\mathbb{R}^n} &&f ( \text{x} ) &\\
  &\text{s.t.} &&c_i ( \text{x} ) \leq 0,&i=1,\cdots,k\\
  &             &&h_j ( \text{x} ) =0,&j=1,\cdots,l
\end{aligned}
$$

引入广义 Lagrange 函数，其中 $\alpha_i,\beta_j$ 是 Language 乘子，并且 $\alpha\geq 0$
$$
L ( \text{x},\alpha,\beta ) =f ( \text{x} ) +\sum_{i=1}^k \alpha_i c_i ( \text{x} ) +\sum_{j=1}^l\beta_j h_j ( \text{x} )
$$

得到 $\text{x}$ 的函数为，其中 $P$ 表示原始问题
$$
\theta_P ( \text{x} ) =\max_{\alpha,\beta: \alpha_i\geq 0} L ( \text{x},\alpha,\beta )
$$

假设某个 $\text{x}$ 违反原始问题的约束条件，即$\exist i, c_i(\text{x})>0$ 或者 $\exist j, h_j ( \text{x} )\neq 0$，令 $\alpha_i\rightarrow + \infty, \beta_j h_j ( \text{x} )\rightarrow +\infty$，其他 $\alpha,\beta$ 均取 $0$ 得
$$
\theta_P( \text{x})=\max_{\alpha,\beta:\alpha_i\geq 0}
  [f( \text{x}) +\sum_{i=1}^k \alpha_i c_i ( \text{x} ) +\sum_{j=1}^l\beta_j h_j ( \text{x} )]
  = +\infty
$$

如果 $\text{x}$ 满足约束条件，则 $\theta_P(\text{x})=f ( \text{x} )$

因此两个极小化问题等价
$$
min_{\text{x}}\theta_P( \text{x})=\min_{\text{x}}\max_{\alpha,\beta:\alpha_i\geq 0} L ( \text{x},\alpha,\beta )
$$

即有相同的解，这样就把原始问题表示为广义 Lagrange函数的极小极大问题。

将原始问题的最优值(解)定义为：$p^*=min_{\text{x}}\theta_P( \text{x})$

## 2. 对偶问题

对偶问题定义：
$$
\theta_D(\alpha,\beta)=min_{\text{x}} L ( \text{x},\alpha,\beta )
$$

再表示为极大化问题
$$
\max_{\alpha,\beta:\alpha_i\geq 0} \theta_D(\alpha,\beta) = \max_{\alpha,\beta:\alpha_i\geq 0} \min_{\text{x}} L ( \text{x},\alpha,\beta )
$$

再将问题转化为最优化问题(原始问题的对偶问题)
$$
\begin{aligned}
  \max_{\alpha,\beta} \theta_D(\alpha,\beta) = \max_{\alpha,\beta} \min_{\text{x}} L ( \text{x},\alpha,\beta )\\
\text{s.t.} \alpha_i\geq 0, i=1,\cdots,k
\end{aligned}
$$

将对偶问题的最优值(解)定义为：$d^*=\max_{\alpha,\beta:\alpha_i\geq 0}\theta_D(\alpha,\beta)$

## 3. 原始问题与对偶问题的关系

定理 C.1：若原始问题和对偶问题都有最优值，则 $d^*=\max_{\alpha,\beta:\alpha_i\geq 0}\min_{\text{x}} L ( \text{x},\alpha,\beta )\leq \min_{\text{x}} max_{\alpha,\beta:\alpha_i\geq 0} L ( \text{x},\alpha,\beta )=p^*$

推论 C.1：设 $x^*$ 和 $\alpha^*,\beta^*$ 分别是原始问题和对偶问题的可行解，并且 $d^*=p^*$，则它们也分别是原始问题和对偶问题的最优解。

定理 C.2：假设函数 $f(\text{x}),c_i(\text{x})$ 是凸函数，$h_j(\text{x})$ 是仿射函数，并且 $\exist \text{x},\forall i, c_i(\text{x})<0$，则存在 $x^*$ 和 $\alpha^*,\beta^*$ ，满足  $x^*$ 是原始问题的解和 $\alpha^*,\beta^*$ 是对偶问题的解，并且 $p^*=d^*=L( \text{x}^*,\alpha^*,\beta^*)$

定理 C.3：假设函数 $f(\text{x}),c_i(\text{x})$ 是凸函数，$h_j(\text{x})$ 是仿射函数，并且 $\exist \text{x},\forall i, c_i(\text{x})<0$，则存在 $x^*$ 和 $\alpha^*,\beta^*$ ，满足  $x^*$ 是原始问题的解和 $\alpha^*,\beta^*$ 是对偶问题的解的充分必要条件是 $x^*$ 和 $\alpha^*,\beta^*$ 满足KKT(Karush-Kuhn-Tucker)条件：
$$
\begin{aligned}
  \nabla_{\text{x}} L(\text{x}^*,\alpha^*,\beta^*)&=0&\\
  \alpha_i^* c_i(\text{x}^*)&=0,&i=1,\cdots,k\\
  c_i(\text{x}^*)&\leq 0,&i=1,\cdots,k\\
  \alpha_i^*&\geq 0,&i=1,\cdots,k\\
  h_j ( \text{x}^*)&=0,&j=1,\cdots,l
\end{aligned}
$$

其中，$\alpha_i^* c_i(\text{x}^*)=0,i=1,\cdots,k$ 是KKT的对偶互补条件，由此条件可知：若 $\alpha_i^*>0$，则 $c_i(\text{x}^*)=0$