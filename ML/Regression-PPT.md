# 回归问题

## 学习提纲

-   根据变量个数分
    -   一元回归
    -   多元回归
-   根据函数类型分
    -   线性模型
    -   线性基函数模型
    -   广义线性模型
-   根据学习方式分
    -   离线学习
    -   在线学习

# 线性无噪声模型

## 一元回归：学习框架

-   函数模型
    -   一个数据对 $( x,y )$
        -   $y = ( 1,x ) ( w_0,w_1 )^T = \dot{\mathbf{x}}^T\mathbf{w}$
        -   $x\in\mathcal{R}, y\in\mathcal{R}, \mathbf{w}\in\mathcal{R}^2, \dot{\mathbf{x}}\in\mathcal{R}^2$
    -   N 个数据对 $( x,y )$
        -   $\mathbf{y} = ( \dot{\mathbf{x}}_1, \dot{\mathbf{x}}_2, \cdots, \dot{\mathbf{x}}_N )^T \mathbf{w} = X \mathbf{w}$
        -   $\mathbf{y}\in\mathcal{R}^N,\mathbf{w}\in\mathcal{R}^2,X\in\mathcal{R}^{N\times 2}$
-   参数求解
    -   欠定问题：无数解
    -   正定问题：惟一解。
        -   $\mathbf{w} = X^{-1} \mathbf{y}$
        -   $\mathbf{y}\in\mathcal{R}^2,\mathbf{w}\in\mathcal{R}^2,X\in\mathcal{R}^{2\times 2}$
    -   超定问题：惟一解。
        -   任取两对数据 $( x_i,y_i )$  即可按正定问题求解。

## 多元回归：学习框架

-   函数模型
    -   一个数据对 $( \mathbf{x},y )$
        -   $y = ( 1,\mathbf{x}^T ) ( w_0,w_1,\cdots,w_D )^T = \dot{\mathbf{x}}^T \mathbf{w}$
        -   $x_d\in\mathcal{R}, y\in\mathcal{R}, \mathbf{w}\in\mathcal{R}^{D+1}, \mathbf{x}\in\mathcal{R}^{D+1}$
    -   N 个数据对 $( \mathbf{x},y )$
        -   $\mathbf{y} = ( \dot{\mathbf{x}}_1, \dot{\mathbf{x}}_2, \cdots, \dot{\mathbf{x}}_N )^T \mathbf{w} = X \mathbf{w}$
        -   $\mathbf{y}\in\mathcal{R}^N,\mathbf{w}\in\mathcal{R}^{D+1},X\in\mathcal{R}^{N \times ( D+1 )}$
-   参数求解
    -   欠定问题：无数解
    -   正定问题：惟一解。
        -   $\mathbf{w} = X^{-1} \mathbf{y}$
        -   $\mathbf{y}\in\mathcal{R}^{D+1}, \mathbf{w}\in\mathcal{R}^{D+1}, X\in\mathcal{R}^{( D+1 ) \times ( D+1 )}$
    -   超定问题：惟一解。
        -   任取 $( D+1 )$  对数据 $( \mathbf{x}_i,y_i )$  即可按正定问题求解。

# 线性有噪声模型

## 一元：有噪声模型：建模

-   线性：无噪声：函数建模
    -   $y = w_0 1 + w_1 x$
    -   $y = ( 1,x ) ( w_0,w_1 )^T = \dot{\mathbf{x}}^T\mathbf{w}$
        -   $y\in\mathcal{R}, x\in\mathcal{R}, \dot{\mathbf{x}}\in\mathcal{R}^2, \mathbf{w}\in\mathcal{R}^2$
-   线性：有噪声：函数建模
    -   $f_\mathbf{w} ( x ) = \dot{\mathbf{x}}^T\mathbf{w}$
    -   $y = f_\mathbf{w} ( x ) + \varepsilon$
        -   $y\in\mathcal{R}, x\in\mathcal{R}, \dot{\mathbf{x}}\in\mathcal{R}^2, \mathbf{w}\in\mathcal{R}^2, \varepsilon\sim\mathcal{N} ( 0,\sigma ) , \sigma\in\mathcal{R}$
-   线性：有噪声：概率建模
    -   $y \sim \mathcal{N} ( f_\mathbf{w} ( x ) ,\sigma )$
        -   $y\in\mathcal{R}, x\in\mathcal{R}, \dot{\mathbf{x}}\in\mathcal{R}^2, \mathbf{w}\in\mathcal{R}^2, \sigma\in\mathcal{R}$

## 多元：有噪声模型：建模

-   线性：无噪声：函数建模
    -   $y = w_0 1 + w_1 x_1 +\cdots + w_D x_D = \sum_{d=0}^D x_d w_d$
    -   $y = ( 1,\mathbf{x}^T ) ( w_0,w_1,\cdots,w_D )^T = \dot{\mathbf{x}}^T \mathbf{w}$
        -   $x_0 = 1, y\in\mathcal{R}, \mathbf{x}\in\mathcal{R}^{D}, \dot{\mathbf{x}}\in\mathcal{R}^{D+1}, \mathbf{w}\in\mathcal{R}^{D+1}$
-   线性：有噪声：函数建模
    -   $f_\mathbf{w} ( \mathbf{x} ) = \dot{\mathbf{x}}^T\mathbf{w}$
    -   $y = f_\mathbf{w} ( \mathbf{x} ) + \varepsilon$
        -   $y\in\mathcal{R}, \mathbf{x}\in\mathcal{R}^{D}, \dot{\mathbf{x}}\in\mathcal{R}^{D+1}, \mathbf{w}\in\mathcal{R}^{D+1}, \varepsilon\sim\mathcal{N} ( 0,\sigma ) , \sigma\in\mathcal{R}$
-   线性：有噪声：概率建模
    -   $y \sim \mathcal{N} ( f_\mathbf{w} ( \mathbf{x} ) ,\sigma )$
        -   $y\in\mathcal{R}, \mathbf{x}\in\mathcal{R}^{D}, \dot{\mathbf{x}}\in\mathcal{R}^{D+1}, \mathbf{w}\in\mathcal{R}^{D+1}, \sigma\in\mathcal{R}$

## 函数建模：学习框架

-   代价函数
    -   平方和误差函数
        -   一元：$J ( \mathbf{w} ) = \frac12\sum_{n=1}^N [y_n- ( {w_0}+{w_1}{x_n} )]^2$
        -   多元：$J ( \mathbf{w} ) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w} ( \mathbf{x}_n )]^2$
    -   正则化 平方和误差函数
        -   多元：$J ( \mathbf{w} ) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w} ( \mathbf{x}_n )]^2+\frac\lambda2||\mathbf{w}||_2$
-   学习方法
    -   微积分：多元函数极值
    -   无约束最优化：最速梯度下降 ( 详情可参考 [^Bishop,2006] [Ch05 Sec 5.2](../PRML/Ch05.md) )
    -   概率统计：最小均方误差估计 / 随机梯度下降

::: notes

最小二乘问题：在不同的模型中代价函数会有不同。使用的思想都是平方误差函数。

计算条件：代价函数 J ( w ) 连续可微。

:::

## 一元：平方和误差函数：多元函数极值

-   代价函数
    -   $J ( \mathbf{w} ) = \frac12 \sum_{n=1}^N [y_n- ( {w_0}+{w_1}{x_n} )]^2$
-   函数求导
    -   $\nabla_{w_0} J ( \mathbf{w} ) =-\sum_{n=1}^N [y_n- ( {w_0}+{w_1}{x_n} )] = 0$
    -   $\nabla_{w_1} J ( \mathbf{w} ) =-\sum_{n=1}^N [y_n- ( {w_0}+{w_1}{x_n} )]{x_n} = 0$
-   参数求解
    -   $\bar{x}=\frac{1}{N}\sum_{n=1}^N{x_n}$
    -   $\bar{y}=\frac{1}{N}\sum_{n=1}^N{y_n}$
    -   $w_1 = \frac{\sum_{n=1}^N {x_n}{y_n} - N\ \bar{x}\ \bar{y}} {\sum_{n=1}^N x_n x_n - N\ \bar{x}\ \bar{x}}$
    -   $w_0=\bar{y}-w_1\bar{x}$

## 多元：平方和误差函数：多元函数极值

-   代价函数
    -   $J ( \mathbf{w} ) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w} ( \mathbf{x}_n )]^2$
-   函数求导
    -   $\nabla_{\mathbf{w}} J ( \mathbf{w} ) = -\sum_{n=1}^N [y_n-f_\mathbf{w} ( \mathbf{x}_n )] \nabla_{\mathbf{w}} ( f_\mathbf{w} ( \mathbf{x}_n )) = \boldsymbol{0}$
-   参数求解
    -   $\bar{x}=\frac{1}{N}\sum_{n=1}^N x_n$
    -   $\bar{y}=\frac{1}{N}\sum_{n=1}^N y_n$
    -   $w_0=\bar{y}-\sum_{d=1}^D w_d\bar{x}$
    -   $\mathbf{w}= ( X^T X )^{-1} X^T \mathbf{y}$

## 正则化平方和误差函数：多元函数极值

-   正则化思想： $E_D ( \mathbf{w} ) +\lambda E_W ( \mathbf{w} )$
    -   λ是正则化系数，$E_D ( \mathbf{w} )$数据依赖误差，$E_W ( \mathbf{w} )$正则化项
-   代价函数
    -   多元：$J ( \mathbf{w} ) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w} ( \mathbf{x}_n )]^2+\frac\lambda2||\mathbf{w}||_q$
        -   $q=1$：Lasso 回归，$||\mathbf{w}||_1$为 L1 范数
        -   $q=2$：Tikhonov 正则，也叫 Ridge 回归。$||\mathbf{w}||_2$为 L2 范数
-   函数求导
    -   $\nabla_{\mathbf{w}} J ( \mathbf{w} ) =-\sum_{n=1}^N [y_n-f_\mathbf{w} ( \mathbf{x}_n )] \nabla_{\mathbf{w}} ( f_\mathbf{w} ( \mathbf{x}_n )) + \frac\lambda2\nabla_{\mathbf{w}} ( ||\mathbf{w}||_2 ) = \boldsymbol{0}$
-   参数求解
    -   $\mathbf{w}= ( \lambda\mathbf{I}+X^TX )^{-1}X^T\mathbf{y}$

::: notes

机器学习：称之为权值衰减

统计学习：称之为 参数收缩

正则化方法通过限制模型的复杂度，避免复杂的模型在有限的数据集上训练时产生过拟合问题。

使得确定最优模型复杂度的问题从确定合适的基函数数量问题转移到确定正则化系数 λ 的值问题。

:::

## 一元：最速梯度下降

-   代价函数
    -   平方和误差函数
        -   $J ( \mathbf{w} ) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w} ( x_n )]^2$
    -   正则化平方和误差函数
        -   $J ( \mathbf{w} ) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w} ( x_n )]^2 + \frac\lambda2||\mathbf{w}||_2$

-   最速梯度下降
    -   $w_0^{\tau+1} = w_0^{\tau} -\eta\nabla_{w_0} J ( \mathbf{w}^{\tau} )$
    -   $w_1^{\tau+1} = w_1^{\tau} -\eta\nabla_{w_1} J ( \mathbf{w}^{\tau} )$
        -   学习率参数： $\eta\in\mathcal{R}$
-   停止条件
    -   当迭代次数 $\tau$  大于 某个值
    -   当 $|\eta\nabla_\mathbf{w} J ( \mathbf{w}^{\tau} ) |$  小于 某个值

## 多元：最速梯度下降

-   代价函数
    -   平方和误差函数
        -   $J ( \mathbf{w} ) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w} ( \mathbf{x}_n )]^2$
    -   正则化平方和误差函数
        -   $J ( \mathbf{w} ) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w} ( \mathbf{x}_n )]^2+\frac\lambda2||\mathbf{w}||_2$
-   最速梯度下降
    -   $\mathbf{w}^{\tau+1} = \mathbf{w}^{\tau} - \eta\nabla_\mathbf{w} J ( \mathbf{w}^{\tau} )$
        -   学习率参数： $\eta\in\mathcal{R}$
        -   与 一元回归 形式相同

-   停止条件
    -   当迭代次数 $\tau$  大于 某个值
    -   当 $|\eta\nabla_\mathbf{w} J ( \mathbf{w}^{\tau} ) |$  小于 某个值

## 一元：最小均方误差 / 随机梯度下降

-   代价函数
    -   平方和误差函数
        -   $J ( \mathbf{w} ) = \frac12 [y_n-f_\mathbf{w} ( x_n )]^2$
    -   正则化平方和误差函数
        -   $J ( \mathbf{w} ) = \frac12 [y_n-f_\mathbf{w} ( x_n )]^2+\frac\lambda2||\mathbf{w}||^2$
-   随机梯度下降
    -   $w_0^{\tau+1} = w_0^{\tau} -\eta\nabla_{w_0} J ( \mathbf{w}^{\tau} )$
    -   $w_1^{\tau+1} = w_1^{\tau} -\eta\nabla_{w_1} J ( \mathbf{w}^{\tau} )$
        -   学习率参数： $\eta\in\mathcal{R}$
-   停止条件
    -   当迭代次数 $\tau$  大于 某个值

::: notes

没有 阈值 作为停止条件，是因为每次只使用其中一个值，而每个值带来的迭代步长是随机的，
因此不能基于 阈值 来判断是否可以停止，而应该基于 迭代次数 来决定。
:::

## 多元：最小均方误差 / 随机梯度下降

-   代价函数
    -   平方和误差函数
        -   $J ( \mathbf{w} ) = \frac12 [y_n-f_\mathbf{w} ( \mathbf{x}_n )]^2$
    -   正则化平方和误差函数
        -   $J ( \mathbf{w} ) = \frac12 [y_n-f_\mathbf{w} ( \mathbf{x}_n )]^2+\frac\lambda2||\mathbf{w}||^2$
-   随机梯度下降
    -   $\mathbf{w}^{\tau+1} = \mathbf{w}^{\tau} - \eta\nabla_\mathbf{w} J ( \mathbf{w}^{\tau} )$
        -   学习率参数： $\eta\in\mathcal{R}$
        -   与 一元回归 形式相同

-   停止条件
    -   当迭代次数 $\tau$  大于 某个值

## 梯度下降算法对比

-   最速梯度下降
    -   在每轮迭代中，选择全部数据优化损失函数
    -   矩阵计算成本高，收敛速度快

-   随机梯度下降，也叫 序列梯度下降
    -   在每轮迭代中，顺序选择一条数据优化损失函数
    -   矩阵计算成本低，收敛速度慢

-   批梯度下降
    -   在每轮迭代中，随机选择一批数据优化损失函数
    -   合理的矩阵计算成本，合理的收敛速度

::: notes

随机梯度下降：极小化代价函数的瞬时值

随机梯度下降收敛性：随机过程中的随机游走 ( 布朗运动 )，收敛于渐近稳定的不动点

:::

## 概率建模：学习框架

-   前提条件
    -   训练样本独立同分布 ( i.i.d. )
    -   训练样本服从高斯分布
    -   高斯分布中的参数稳定不变
-   建立模型
    -   无噪声函数：$f_\mathbf{w} ( \mathbf{x} ) = \dot{\mathbf{x}}^T\mathbf{w}$
    -   有噪声模型：概率建模：$y \sim \mathcal{N} ( f_\mathbf{w} ( \mathbf{x} ) ,\sigma )$
        -   $y\in\mathcal{R}, \mathbf{w}\in\mathcal{R}^{D+1}, \mathbf{x}\in\mathcal{R}^{D}, \dot{\mathbf{x}}\in\mathcal{R}^{D+1}$
-   代价函数
    -   离线学习
        -   最大似然估计
        -   最大后验估计
    -   在线学习：最小均方误差

::: notes

训练样本存在相关性，就需要更加复杂的随机过程去构建模型。

因为高斯分布的概率密度函数计算方便，如果是其他概率密度函数就需要考虑其他的方法来计算了。

参数如果是变化的，那么就需要在线学习，也称自适应算法。

与 Markov 过程 和 朴素贝叶斯分类 的区别

-   最大似然估计，每个样本相互之间独立同分布
-   Markov 过程，每个样本只与它附近的样本相关
-   朴素贝叶斯分类，每个特征之间相互独立

:::

## 最大似然估计

-   似然函数
    -   $p ( \mathbf{y}|\mathbf{x},\mathbf{w},\sigma ) =\prod_{n=1}^N\mathcal{N} ( y_n|f_{\mathbf{w}} ( \mathbf{x}_n ) ,\sigma^2 )$
    -   $\text{ln } p ( \mathbf{y}|\mathbf{x},\mathbf{w},\sigma ) = -\frac{1}{2\sigma^2} \sum_{n=1}^N [y_n - f_{\mathbf{w}} ( \mathbf{x}_n )]^2 + \frac{N}{2}\text{ln } ( \sigma^2 ) -\frac{N}{2}\text{ln } ( 2\pi )$
-   参数求解
    -   $\mathbf{w}_{{}_{ML}}=\text{arg min}_{\mathbf{w}}\text{ln } p ( \mathbf{y}|\mathbf{x,w},\sigma )$
    -   $\mathbf{w}_{{}_{ML}}\propto\text{arg min}_{\mathbf{w}}\sum_{n=1}^N [y_n-f_{\mathbf{w}} ( \mathbf{x}_n )]^2$
-   学习方法
    -   训练样本服从高斯分布时 等价于  "平方误差函数" 的极值

::: notes

增强对 平方误差函数 的理解

:::

## 偏差——方差分解

-   偏差：度量的是匹配的"准确性"和"质量"。
    -   高的偏差 表示 坏的匹配
-   方差：度量的是匹配的"精确性"和"特定性"。
    -   高的方差 表示 弱的匹配
-   期望损失 = ( 偏差 )^2^ + 方差 + 噪声
    -   ( 偏差 )^2^ = $\int\{\mathbf{E}_\mathcal{D}[f ( \text{x};\mathcal{D} ) -h ( \text{x} )]\}^2 p ( \text{x} ) \text{ dx}$
    -   方差 = $\int\{\mathbf{E}_\mathcal{D}[f ( \text{x};\mathcal{D} ) -\mathbf{E}_{\mathcal{D}}[f ( \text{x};\mathcal{D} )] ) \}^2] p ( \text{x} ) \text{ dx}$
    -   噪声 = $\int\int [h ( \text{x} ) - y]^2 p ( \text{x},y ) \text{dx d}y$
        -   预测模型：$f ( \text{x};\mathcal{D} )$
        -   最优模型：$h ( \text{x} ) = \mathbf{E}[y|\text{x}] = \int y p ( y|\text{x} ) \text{ d}y$
        -   目标数据：$y$

注：偏差——方差分解属于频率论的角度，依赖于对所有的数据集求平均。[^Bishop,2006] Sec.3.2. [^Duda,2003] Sec.9.3.

## 最大后验估计

-   后验概率
    -   先验分布：$p ( \mathbf{w}|\sigma_\mathbf{w} ) = \mathcal{N} ( \mathbf{w}|\boldsymbol{0},\sigma_w^2\mathbf{I} )$
    -   似然函数：$p ( \mathbf{y}|\mathbf{x},\mathbf{w},\sigma ) = \prod_{n=1}^N \mathcal{N} ( y_n|f_{\mathbf{w}} ( \mathbf{x}_n ) ,\sigma^2 )$
    -   后验分布：$p ( \mathbf{w}|y,\mathbf{x},\sigma ) \propto p ( y|\mathbf{x},\mathbf{w},\sigma ) p ( \mathbf{w}|\sigma_\mathbf{w} )$
-   参数求解
    -   $\mathbf{w}_{{}_{MAP}}=\text{arg min}_{\mathbf{w}} \text{ln } p ( \mathbf{w}|y,\mathbf{x},\sigma )$
    -   $\mathbf{w}_{{}_{MAP}}\propto\text{arg min}_{\mathbf{w}} ( \sum_{n=1}^N [y_n-f_{\mathbf{w}} ( x_n )]^2 + \frac{\lambda}{2}||\mathbf{w}||_2 )$
        -   $\lambda= ( \frac{\sigma}{\sigma_w} )^2$
-   学习方法
    -   训练样本和先验服从高斯分布时 等价于 "正则化平方误差函数" 的极值

::: notes

增强对 正则化平方误差函数 的理解

-   为什么会有正则化？
-   如何确定正则化系数 λ？
-   如何加入先验信息？

:::

# 线性基函数模型

## 线性基函数建模

-   线性：无噪声：函数建模
    -   $y = w_0 1 + w_1 x_1 +\cdots + w_D x_D = \sum_{d=0}^D x_d w_d$
    -   $y = ( 1,\mathbf{x}^T ) ( w_0,w_1,\cdots,w_D )^T = \dot{\mathbf{x}}^T \mathbf{w}$
        -   $x_0 = 1, y\in\mathcal{R}, \mathbf{x}\in\mathcal{R}^{D}, \dot{\mathbf{x}}\in\mathcal{R}^{D+1}, \mathbf{w}\in\mathcal{R}^{D+1}$
-   线性基函数：有噪声：函数建模
    -   $f_\mathbf{w} ( \mathbf{x} ) = w_0 1+w_1 g_1 ( \mathbf{x} ) +\dots+w_{{}_M} g_{{}_M} ( \mathbf{x} ) = \sum_{m=0}^M g_m ( \mathbf{x} ) w_m$
    -   $f_\mathbf{w} ( \mathbf{x} ) = ( 1,\mathbf{g} ( \mathbf{x} )) \mathbf{w} = \dot{\mathbf{g}} ( \mathbf{x} )^T\mathbf{w}$
    -   $y =f_\mathbf{w} ( \mathbf{x} ) + \varepsilon$
        -   $g_0 ( \mathbf{x} ) = 1, y\in\mathcal{R}, \mathbf{x}\in\mathcal{R}^{D}, \dot{\mathbf{g}} ( \cdot ) \in\mathcal{R}^{M+1}, \mathbf{w}\in\mathcal{R}^{M+1}, \varepsilon\sim\mathcal{N} ( 0,\sigma ) , \sigma\in\mathcal{R}$
-   线性基函数：有噪声：概率建模
    -   $y \sim \mathcal{N} ( f_\mathbf{w} ( \mathbf{x} ) ,\sigma )$
        -   $y\in\mathcal{R}, \mathbf{x}\in\mathcal{R}^{D}, \dot{\mathbf{g}} ( \cdot ) \in\mathcal{R}^{M+1}, \mathbf{w}\in\mathcal{R}^{M+1}, \sigma\in\mathcal{R}$

## 基函数的选择

-   多项式函数： $g_m ( x ) = x^m$
    -   线性无关：$k_1=k_2=k_3=0, k_1 f_1 ( \cdot ) + k_2 f_2 ( \cdot ) + k_3 f_3 ( \cdot ) =0$
-   高斯函数： $g_m ( x ) = \exp[-\frac{( x-\mu_m )^2}{2\sigma^2}]$
-   sigmoid 函数： $g_m ( x ) = \varsigma ( \frac{x-\mu_m}{\sigma} )$
    -   logistic sigmoid 函数：$\varsigma ( a ) =\frac1{1+\exp ( -a )}$
-   Fourier 函数
    -   规范正交： $f_i ( \cdot ) f_j ( \cdot ) =\begin{cases} 1 & i=j\\0, & i\neq j\end{cases}$
-   Wavelet 函数
    -   规范正交

::: notes

可以把多项式基函数模型看作线性模型的数据经过非线性特征变换，变换后的特征线性无关。

可以把 Fourier 基函数模型看作线性模型的数据经过非线性特征变换，变换后的特征规范正交。

:::

## 函数建模：学习框架

-   代价函数
    -   平方和误差函数 ( 一元与多元运算没有区别 )
        -   一元：$J ( \mathbf{w} ) = \frac12\sum_{n=1}^N [y_n-\sum_{m=0}^M g_m ( x_n ) w_m]^2$
        -   多元：$J ( \mathbf{w} ) = \frac12\sum_{n=1}^N [y_n-\sum_{m=0}^M g_m ( \mathbf{x}_n ) w_m]^2$
    -   正则化 平方和误差函数
        -   多元：$J ( \mathbf{w} ) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w} ( \mathbf{x}_n )]^2+\frac\lambda2||\mathbf{w}||_2$
-   学习方法
    -   多元函数极值 ( 微积分 )
    -   最速梯度下降 ( 无约束最优化 )
    -   最小均方误差估计 / 随机梯度下降 ( 概率统计 )

## 平方和误差函数：多元函数极值

-   代价函数
    -   $J ( \mathbf{w} ) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w} ( \mathbf{x}_n )]^2+\frac\lambda2||\mathbf{w}||_2$
-   函数求导
    -   $\nabla_{\mathbf{w}} J ( \mathbf{w} ) = -\sum_{n=1}^N [y_n-f_\mathbf{w} ( \mathbf{x}_n )] \nabla_{\mathbf{w}} ( f_\mathbf{w} ( \mathbf{x}_n )) = \boldsymbol{0}$
-   参数求解
    -   $w_0=\bar{y}-\sum_{m=1}^M w_m \bar{g}_m ( \mathbf{x}_n )$
        -   $\bar{g}_m ( \mathbf{x}_n ) =\frac{1}{N}\sum_{n=1}^N g_m ( \mathbf{x}_n )$ , $\bar{y}=\frac{1}{N}\sum_{n=1}^N y_n$
    -   $\mathbf{w}= ( G^T G )^{-1} G^T \mathbf{y} = G^{\dagger}\mathbf{y}$
        -   伪逆矩阵：$G^{\dagger} \equiv ( G^T G )^{-1} G^T$
        -   设计矩阵：$G=\begin{bmatrix} g_0 ( \mathbf{x}_1 ) & g_1 ( \mathbf{x}_1 ) & \cdots & g_{{}_M} ( \mathbf{x}_1 ) \\ g_0 ( \mathbf{x}_2 ) & g_1 ( \mathbf{x}_2 ) & \cdots & g_{{}_M} ( \mathbf{x}_2 ) \\ \vdots & \vdots & \ddots & \vdots \\ g_0 ( \mathbf{x}_{{}_N} ) & g_1 ( \mathbf{x}_{{}_N} ) & \cdots & g_{{}_M} ( \mathbf{x}_{{}_N} ) \\ \end{bmatrix}$

## 正则化平方和误差函数：多元函数极值

-   正则化思想： $E_D ( \mathbf{w} ) +\lambda E_W ( \mathbf{w} )$
    -   λ是正则化系数，$E_D ( \mathbf{w} )$数据依赖误差，$E_W ( \mathbf{w} )$正则化项
-   代价函数
    -   多元：$J ( \mathbf{w} ) = \frac12\sum_{n=1}^N [y_n-\sum_{m=0}^M g_m ( \mathbf{x}_n ) w_m]^2+\frac\lambda2||\mathbf{w}||_q$
        -   $q=1$：Lasso 回归，$||\mathbf{w}||_1$为 L1 范数
        -   $q=2$：Tikhonov 正则，也叫 Ridge 回归。$||\mathbf{w}||_2$为 L2 范数
-   函数求导
    -   $\nabla_{\mathbf{w}} J ( \mathbf{w} ) =-\sum_{n=1}^N [y_n-f_\mathbf{w} ( \mathbf{x}_n )] \nabla_{\mathbf{w}} ( f_\mathbf{w} ( \mathbf{x}_n )) + \frac\lambda2\nabla_{\mathbf{w}} ( ||\mathbf{w}||_2 ) = \boldsymbol{0}$
-   参数求解
    -   $\mathbf{w}= ( \lambda\mathbf{I}+G^T G )^{-1} G^T \mathbf{y} = G^{\dagger}\mathbf{y}$

## 基函数：最大似然估计

-   似然函数
    -   $p ( \mathbf{y}|\mathbf{x},\mathbf{w},\sigma ) =\prod_{n=1}^N\mathcal{N} ( y_n|f_{\mathbf{w}} ( \mathbf{x}_n ) ,\sigma^2 )$
    -   $\text{ln } p ( \mathbf{y}|\mathbf{x},\mathbf{w},\sigma ) = -\frac{1}{2\sigma^2} \sum_{n=1}^N [y_n - f_{\mathbf{w}} ( \mathbf{x}_n )]^2 + \frac{N}{2}\text{ln } ( \sigma^2 ) -\frac{N}{2}\text{ln } ( 2\pi )$
-   参数求解
    -   $\mathbf{w}_{{}_{ML}}=\text{arg min}_{\mathbf{w}}\text{ln } p ( \mathbf{y}|\mathbf{x,w},\sigma )$
    -   $\mathbf{w}_{{}_{ML}}\propto\text{arg min}_{\mathbf{w}}\sum_{n=1}^N [y_n-f_{\mathbf{w}} ( \mathbf{x}_n )]^2$
-   学习方法
    -   训练样本服从高斯分布时 等价于  "平方误差函数" 的极值

## 基函数：最大后验估计

-   后验概率
    -   先验分布：$p ( \mathbf{w}|\sigma_\mathbf{w} ) = \mathcal{N} ( \mathbf{w}|\boldsymbol{0},\sigma_w^2\mathbf{I} )$
    -   似然函数：$p ( \mathbf{y}|\mathbf{x},\mathbf{w},\sigma ) = \prod_{n=1}^N \mathcal{N} ( y_n|f_{\mathbf{w}} ( \mathbf{x}_n ) ,\sigma^2 )$
    -   后验分布：$p ( \mathbf{w}|y,\mathbf{x},\sigma ) \propto p ( y|\mathbf{x},\mathbf{w},\sigma ) p ( \mathbf{w}|\sigma_\mathbf{w} )$
-   参数求解
    -   $\mathbf{w}_{{}_{MAP}}=\text{arg min}_{\mathbf{w}} \text{ln } p ( \mathbf{w}|y,\mathbf{x},\sigma )$
    -   $\mathbf{w}_{{}_{MAP}}\propto\text{arg min}_{\mathbf{w}} ( \sum_{n=1}^N [y_n-f_{\mathbf{w}} ( x_n )]^2 + \frac{\lambda}{2}||\mathbf{w}||_2 )$
        -   $\lambda= ( \frac{\sigma}{\sigma_w} )^2$
-   学习方法
    -   训练样本和先验服从高斯分布时 等价于 "正则化平方误差函数" 的极值

# 广义线性模型

## 一元回归：广义线性模型

-   线性无噪声模型
    -   $y = w_0 1 + w_1 x = \mathbf{w}^T\mathbf{x}$
        -   $x\in\mathcal{R}, y\in\mathcal{R}$
-   广义线性模型
    -   $y = g ( \mathbf{w}^T\mathbf{x} )$
        -   $x\in\mathcal{R}, y\in\mathcal{R}$
