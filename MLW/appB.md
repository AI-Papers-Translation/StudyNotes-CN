# 附录 B

## B.1 Lagrange Multipliers

Lagrange 乘子法用于寻找多元函数在一组约束下的极值，通过引入 Language 乘子，可以将有 $d$ 个变量与 $k$ 个约束条件的最优化问题转化为具有 $d+k$ 个变量的无约束优化问题进行求解

一个约束的问题求解

-   等式约束的优化问题
    -   假定 $\text{x}\in\mathbb{R}^{D}$，欲寻找 $\text{x}$ 的某个取值 $\text{x}^*$ 使目标函数 $f ( \text{x} )$ 最小，并且同时满足 $g ( \text{x} ) =0$ 的约束需要。
    -   从几何角度看，就是在方程 $g ( \text{x} ) =0$ 确定的 $d-1$ 维约束曲面上寻找能使目标函数 $f ( \text{x} )$ 最小化的点
        -   对于约束曲面上的任意点 $\text{x}$，该点的梯度 $\nabla g ( \text{x} )$ 正交于约束曲面
        -   在最优点 $\text{x}^*$，该点的梯度 $\nabla f ( \text{x}^* )$ 正交于约束曲面

-   不等式约束的优化问题
    -   假定 $\text{x}\in\mathbb{R}^{D}$，欲寻找 $\text{x}$ 的某个取值 $\text{x}^*$ 使目标函数 $f ( \text{x} )$ 最小，并且同时满足 $g ( \text{x} ) \leq 0$ 的约束需要。
    -   转化为 KKT ( Karush-Kuhn-Tucker ) 条件

$$
\begin{cases}
    g ( \text{x} ) \leq 0\\
    \lambda\geq 0\\
    \lambda g ( \text{x} ) =0
\end{cases}
$$

多个约束的问题求解

一个优化问题可以从两个角度来考察

-   主问题(Primal Problem)
-   对偶问题(Dual Problem)

-   主问题描述：$m$ 个等式约束 和 $n$ 个不等式约束，可行域 $\mathbb{D}\sub\mathbb{R}^{d}$ 非空

$$
\begin{aligned}
    & \min_{\text{x}} f ( \text{x} ) \\
    \text{s.t.}
        & h_i ( \text{x} ) =0 ( i=1,\cdots,m ) \\
        & g_j ( \text{x} ) \leq 0 ( j=1,\cdots,n ) 
\end{aligned}
$$

-   引入 Lagrange 乘子 $\boldsymbol{\lambda}=(\lambda_1,\cdots,\lambda_m)^T$ 和 $\boldsymbol{\mu}=(\mu_1,\cdots,\mu_n)^T$，则 Lagrange 函数为

$$
L ( \text{x},\boldsymbol{\lambda},\boldsymbol{\mu} )=f ( \text{x} ) +\sum_{i = 1}^m\lambda_i h_i ( \text{x} ) +\sum_{j=1}^n \mu_j g_j ( \text{x} )
$$

-   由不等式约束引入的 KKT 条件 $(j=1,\cdots,n)$ 为

$$
\begin{cases}
    g_j ( \text{x} ) \leq 0\\
    \mu_j\geq 0\\
    \mu_j g_j ( \text{x} ) =0
\end{cases}
$$

-   对偶问题描述
    -   对偶函数 $\Gamma : \mathbb{R}\times\mathbb{R}\mapsto\mathbb{R}$ 定义为 [^1] $\Gamma(\boldsymbol{\lambda},\boldsymbol{\mu})=\inf_{\text{x}\in\mathbb{D}} L( \text{x},\boldsymbol{\lambda},\boldsymbol{\mu})$
    -   若 $\tilde{\text{x}}\in\mathbb{D}$ 为主问题可行域中的点，则对任意 $\boldsymbol{\mu}\succeq 0$ 和 $\boldsymbol{\lambda}$ 都有 $\sum_{i = 1}^m\lambda_i h_i ( \text{x} ) +\sum_{j=1}^n \mu_j g_j ( \text{x} )\leq 0$
        -   可得：$\Gamma(\boldsymbol{\lambda},\boldsymbol{\mu})=\inf_{\text{x}\in\mathbb{D}} L( \text{x},\boldsymbol{\lambda},\boldsymbol{\mu})\leq L( \tilde{\text{x}},\boldsymbol{\lambda},\boldsymbol{\mu})\leq f( \tilde{\text{x}})$
    -   若 $p^*$ 为主问题的最优值，则对任意 $\boldsymbol{\mu}\succeq 0$ 和 $\boldsymbol{\lambda}$ 都有：$\Gamma(\boldsymbol{\lambda},\boldsymbol{\mu})\leq p^*$
    -   对偶函数给出了主问题最优值的下界，因此如何寻找最好的下界 $\max_{\boldsymbol{\lambda},\boldsymbol{\mu}} \Gamma(\boldsymbol{\lambda},\boldsymbol{\mu}) \text{ s.t. } \boldsymbol{\mu}\succeq 0$ 就是主问题的对偶问题
        -   $\boldsymbol{\lambda}$ 和 $\boldsymbol{\mu}$ 称为「对偶变量」
    -   无论主问题是否为凸函数，对偶问题始终是凸函数
    -   对偶问题的对偶性
        -   「弱对偶性」：对偶问题的最优值 $d^*\leq p^*$
        -   「强对偶性」：对偶问题的最优值 $d^*=p^*$

[^1]:inf 表示下确界

<!--TODO:最优化问题还需要补充-->