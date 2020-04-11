# 回归问题

## 学习提纲

- 根据变量个数分
  - 一元回归
  - 多元回归

- 根据函数类型分
  - 线性模型
  - 线性基函数模型
  - 广义线性模型

- 根据学习方式分
  - 离线学习
  - 在线学习

# 线性无噪声模型

## 一元回归：学习框架

- 函数模型
  - 一个数据对 $(x,y)$： $y = (1,x) (w_0,w_1)^T = \dot{\mathbf{x}}^T\mathbf{w}$
    - $x\in\mathcal{R}, y\in\mathcal{R}, \mathbf{w}\in\mathcal{R}^2, \dot{\mathbf{x}}\in\mathcal{R}^2$
  - N个数据对 $(x,y)$： $\mathbf{y} = (\dot{\mathbf{x}}_1, \dot{\mathbf{x}}_2, \cdots, \dot{\mathbf{x}}_N)^T \mathbf{w} = X \mathbf{w}$
    - $\mathbf{y}\in\mathcal{R}^N,\mathbf{w}\in\mathcal{R}^2,X\in\mathcal{R}^{N\times 2}$

- 参数求解
  - 欠定问题：无数解
  - 正定问题：惟一解。
    - $\mathbf{w} = X^{-1} \mathbf{y}$  ，
      - $\mathbf{y}\in\mathcal{R}^2,\mathbf{w}\in\mathcal{R}^2,X\in\mathcal{R}^{2\times 2}$
  - 超定问题：惟一解。
    - 任取两对数据 $(x_i,y_i)$  即可按正定问题求解。

## 多元回归：学习框架

- 函数模型
  - 一个数据对 $(\mathbf{x},y)$：$y = (1,\mathbf{x}^T)(w_0,w_1,\cdots,w_D)^T = \dot{\mathbf{x}}^T \mathbf{w}$
    - $x_d\in\mathcal{R}, y\in\mathcal{R}, \mathbf{w}\in\mathcal{R}^{D+1}, \mathbf{x}\in\mathcal{R}^{D+1}$
  - N个数据对 $(\mathbf{x},y)$：$\mathbf{y} = (\dot{\mathbf{x}}_1, \dot{\mathbf{x}}_2, \cdots, \dot{\mathbf{x}}_N)^T \mathbf{w} = X \mathbf{w}$
    - $\mathbf{y}\in\mathcal{R}^N,\mathbf{w}\in\mathcal{R}^{D+1},X\in\mathcal{R}^{N \times (D+1)}$

- 参数求解
  - 欠定问题：无数解
  - 正定问题：惟一解。
    - $\mathbf{w} = X^{-1} \mathbf{y}$
      - $\mathbf{y}\in\mathcal{R}^{D+1}, \mathbf{w}\in\mathcal{R}^{D+1}, X\in\mathcal{R}^{(D+1)\times (D+1)}$
  - 超定问题：惟一解。
    - 任取 $(D+1)$  对数据 $(\mathbf{x}_i,y_i)$  即可按正定问题求解。

# 线性有噪声模型

## 一元回归：有噪声模型建模

- 线性无噪声模型
  - 函数建模
    - $y = w_0 1 + w_1 x = (1,x) (w_0,w_1)^T = \dot{\mathbf{x}}^T\mathbf{w}$
      - $x\in\mathcal{R}, y\in\mathcal{R}, \mathbf{w}\in\mathcal{R}^2, \dot{\mathbf{x}}\in\mathcal{R}^2$

- 线性带噪声模型
  - 无噪声函数
    - $f_\mathbf{w}(\dot{\mathbf{x}}) = \dot{\mathbf{x}}^T\mathbf{w}$

  - 函数建模
    - $y = f_\mathbf{w}(\dot{\mathbf{x}}) + \varepsilon$
      - $y\in\mathcal{R}, \mathbf{w}\in\mathcal{R}^2, \dot{\mathbf{x}}\in\mathcal{R}^2, \varepsilon\sim\mathcal{N}(0,\sigma)$

  - 概率建模
    - $y \sim \mathcal{N}(f_\mathbf{w}(\dot{\mathbf{x}}),\sigma)$

## 函数建模：学习框架

- 代价函数
  - 平方和误差函数：
    - $J(\mathbf{w}) = \frac12\sum_{n=1}^N[y_n-({w_0}+{w_1}{x_n})]^2 = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w}(x_n)]^2$
  - 正则化平方和误差函数：
    - $J(\mathbf{w}) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w}(x_n)]^2+\frac\lambda2||\mathbf{w}||^2$
  
- 学习方法
  - 多元函数极值（微积分）
  - 最速梯度下降（无约束最优化）
  - 最小均方误差估计 / 随机梯度下降（概率统计）

::: notes

- 最小二乘问题：在不同的模型中代价函数会有不同。使用的思想都是平方误差函数。

- 计算条件：代价函数 J(w) 连续可微。

:::

## 平方和误差函数：多元函数极值

- 代价函数
  - $J(\mathbf{w}) = \frac12 \sum_{n=1}^N[y_n-({w_0}+{w_1}{x_n})]^2$

- 函数求导
  - $\nabla_{w_0} J(\mathbf{w})=-\sum_{n=1}^N[y_n-({w_0}+{w_1}{x_n})] = 0$
  - $\nabla_{w_1} J(\mathbf{w})=-\sum_{n=1}^N[y_n-({w_0}+{w_1}{x_n})]{x_n} = 0$

- 解
  - $w_1^* = \frac{\sum_{n=1}^N {x_n}{y_n} - N\bar{x}\bar{y}} {\sum_{n=1}^N x_n x_n - N\bar{x}\bar{x}}$  , $w_0^*=\bar{y}-w_1^*\bar{x}$
  - $\bar{x}=\frac{1}{N}\sum_{n=1}^N{x_n}$  , $\bar{y}=\frac{1}{N}\sum_{n=1}^N{y_n}$

## 正则化平方和误差函数：多元函数极值

- 正则化思想： $E_D(\mathbf{w})+\lambda E_W(\mathbf{w})$
  - λ 是正则化系数， $E_D(\mathbf{w})$  数据依赖误差， $E_W(\mathbf{w})$  正则化项

- 代价函数
  - $J(\mathbf{w}) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w}(x_n)]^2+\frac\lambda2||\mathbf{w}||^q$
    - $q=1$：Lasso 回归
    - $q=2$：Tikhonov 正则，也叫 Ridge 回归

- 函数求导
  - $\nabla_{\mathbf{w}} J(\mathbf{w})=-\sum_{n=1}^N[y_n-f_\mathbf{w}(x_n)] \nabla_{\mathbf{w}}(f_\mathbf{w}(x_n))+ \frac\lambda2\nabla_{\mathbf{w}}(||\mathbf{w}||^2)= \boldsymbol{0}$
  - $\mathbf{w}=(\lambda\mathbf{I}+X^TX)^{-1}X^T\mathbf{y}$
  
## 最速梯度下降

- 代价函数
  - 平方和误差函数：$J(\mathbf{w}) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w}(x_n)]^2$
  - 正则化平方和误差函数：$J(\mathbf{w}) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w}(x_n)]^2 + \frac\lambda2||\mathbf{w}||^2$

- 最速梯度下降
  - $w_0^{\tau+1} = w_0^{\tau} -\eta\nabla_{w_0} J(\mathbf{w}^{\tau})$
  - $w_1^{\tau+1} = w_1^{\tau} -\eta\nabla_{w_1} J(\mathbf{w}^{\tau})$
  - 学习率参数： $\eta\in\mathcal{R}$

- 停止条件
  - 当迭代次数 $\tau$  大于 某个值
  - 当 $|\eta\nabla_\mathbf{w} J(\mathbf{w}^{\tau})|$  小于 某个值

## 最小均方误差 / 随机梯度下降

- 代价函数
  - 平方和误差函数：
    - $J(\mathbf{w}) = \frac12 [y_n-f_\mathbf{w}(x_n)]^2$
  - 正则化平方和误差函数：
    - $J(\mathbf{w}) = \frac12 [y_n-f_\mathbf{w}(x_n)]^2+\frac\lambda2||\mathbf{w}||^2$

- 随机梯度下降：极小化代价函数的瞬时值
  - $w_0^{\tau+1} = w_0^{\tau} -\eta\nabla_{w_0} J(\mathbf{w}^{\tau})$
  - $w_1^{\tau+1} = w_1^{\tau} -\eta\nabla_{w_1} J(\mathbf{w}^{\tau})$
    - 学习率参数： $\eta\in\mathcal{R}$

- 停止条件
  - 当迭代次数 $\tau$  大于 某个值
  - 当 $|\eta\nabla_\mathbf{w} J(\mathbf{w}^{\tau})|$  小于 某个值

## 梯度下降算法对比

- 最速梯度下降
  - 在每轮迭代中，选择全部数据优化损失函数
  - 矩阵计算成本高， 收敛速度快
  
- 随机梯度下降，也叫 序列梯度下降
  - 在每轮迭代中，顺序选择一条数据优化损失函数
  - 矩阵计算成本低， 收敛速度慢
  
- 批梯度下降
  - 在每轮迭代中，随机选择一批数据优化损失函数
  - 合理的矩阵计算成本，合理的收敛速度

::: notes

- 随机梯度下降：极小化代价函数的瞬时值
- 随机梯度下降收敛性：随机过程中的随机游走（布朗运动），收敛于渐近稳定的不动点

:::

## 概率建模：学习框架

- 线性有噪声模型
  - 无噪声函数：$f_\mathbf{w}(\dot{\mathbf{x}}) = \dot{\mathbf{x}}^T\mathbf{w}$
  - 概率建模：$y \sim \mathcal{N}(f_\mathbf{w}(\dot{\mathbf{x}}),\sigma)$  ， $y\in\mathcal{R}, \mathbf{w}\in\mathcal{R}^2, \dot{\mathbf{x}}\in\mathcal{R}^2$

- 代价函数
  - 离线学习
    - 最大似然估计
    - 最大后验估计
  - 在线学习：最小均方误差

## 最大似然估计

- 似然函数
  - $p(\mathbf{y}|\mathbf{x},\mathbf{w},\sigma)=\prod_{n=1}^N\mathcal{N}(y_n|f_{\mathbf{w}}(x_n),\sigma^2)$
  - $\ln p(\mathbf{y}|\mathbf{x},\mathbf{w},\sigma) = -\frac{1}{2\sigma^2} \sum_{n=1}^N [y_n - f_{\mathbf{w}}(x_n)]^2 + \frac{N}{2}\ln(\sigma^2)-\frac{N}{2}\ln(2\pi)$

- 解
  - $\mathbf{w}_{_{ML}}=arg min_{\mathbf{w}}\ln p(\mathbf{y}|\mathbf{x,w},\sigma)$
  - $\mathbf{w}_{ML}\propto\arg\min_{\mathbf{w}}\sum_{n=1}^N[y_n-f_{\mathbf{w}}(x_n)]^2$
  
- 学习方法
  - 等价于  "平方误差函数" 的极值

## 最大后验估计

- 后验概率
  - $p(\mathbf{w}|\sigma_\mathbf{w}) = \mathcal{N}(\mathbf{w}|\boldsymbol{0},\sigma_w^2\mathbf{I})$
  - $p(\mathbf{w}|y,\mathbf{x},\sigma)\propto p(y|\mathbf{x},\mathbf{w},\sigma) p(\mathbf{w}|\sigma_\mathbf{w})$

- 解
  - $\mathbf{w}_{_{MAP}}=\arg\min_{\mathbf{w}} \ln p(\mathbf{w}|y,\mathbf{x},\sigma)\propto\arg\min_{\mathbf{w}}\sum_{n=1}^N[y_n-f_{\mathbf{w}}(x_n)]^2 + \frac{\lambda}{2}||\mathbf{w}||^2$
    - $\lambda=(\frac{\sigma}{\sigma_w})^2$

- 学习方法
  - 等价于 “正则化平方误差函数” 的极值

## 多元回归：有噪声模型建模

- 线性无噪声模型
  - 函数建模
    - $y = (1,\mathbf{x}^T)(w_0,w_1,\cdots,w_D)^T = \dot{\mathbf{x}}^T \mathbf{w}$
      - $\mathbf{y}\in\mathcal{R},\mathbf{w}\in\mathcal{R}^{D+1},\dot{\mathbf{w}}\in\mathcal{R}^{D+1}$

- 线性带噪声模型
  - 无噪声函数
    - $f_\mathbf{w}(\dot{\mathbf{x}}) = \dot{\mathbf{x}}^T\mathbf{w}$

  - 函数建模
    - $y = f_\mathbf{w}(\dot{\mathbf{x}}) + \varepsilon$
      - $y\in\mathcal{R}, \mathbf{w}\in\mathcal{R}^{D+1}, \dot{\mathbf{x}}\in\mathcal{R}^{D+1}, \varepsilon\sim\mathcal{N}(0,\sigma)$

  - 概率建模
    - $y \sim \mathcal{N}(f_\mathbf{w}(\dot{\mathbf{x}}),\sigma)$

## 函数建模：学习框架

- 代价函数
  - 平方和误差函数：
    - $J(\mathbf{w}) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w}(\mathbf{x}_n)]^2$
  - 正则化平方和误差函数：
    - $J(\mathbf{w}) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w}(\mathbf{x}_n)]^2+\frac\lambda2||\mathbf{w}||^2$
  
- 学习方法
  - 多元函数极值（微积分）
  - 最速梯度下降（无约束最优化）
  - 最小均方误差估计 / 随机梯度下降（概率统计）

::: notes

- 最小二乘问题：在不同的模型中代价函数会有不同。使用的思想都是平方误差函数。

- 计算条件：代价函数 J(w) 连续可微。

:::

## 平方和误差函数：多元函数极值

- 代价函数
  - $J(\mathbf{w}) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w}(\mathbf{x}_n)]^2$

- 函数求导
  - $\nabla_{\mathbf{w}} J(\mathbf{w}) = -\sum_{n=1}^N [y_n-f_\mathbf{w}(\mathbf{x}_n)] \nabla_{\mathbf{w}} (f_\mathbf{w}(\mathbf{x}_n)) = 0$

- 解
  - $\mathbf{w}=(X^TX)^{-1}X^T\mathbf{y}$

## 正则化平方和误差函数：多元函数极值

- 正则化思想： $E_D(\mathbf{w})+\lambda E_W(\mathbf{w})$
  - λ 是正则化系数， $E_D(\mathbf{w})$  数据依赖误差， $E_W(\mathbf{w})$  正则化项

- 代价函数
  - $J(\mathbf{w}) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w}(\mathbf{x}_n)]^2+\frac\lambda2||\mathbf{w}||^q$
    - $q=1$：Lasso 回归
    - $q=2$：Tikhonov 正则，也叫 Ridge 回归

- 函数求导
  - $\nabla_{\mathbf{w}} J(\mathbf{w})=-\sum_{n=1}^N[y_n-f_\mathbf{w}(\mathbf{x}_n)] \nabla_{\mathbf{w}}(f_\mathbf{w}(\mathbf{x}_n))+ \frac\lambda2\nabla_{\mathbf{w}}(||\mathbf{w}||^2)= \boldsymbol{0}$
- 解
  - $\mathbf{w}=(\lambda\mathbf{I}+X^TX)^{-1}X^T\mathbf{y}$
  - 与 一元回归 形式相同

## 最速梯度下降

- 代价函数
  - 平方和误差函数：
    - $J(\mathbf{w}) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w}(\mathbf{x}_n)]^2$
  - 正则化平方和误差函数：
    - $J(\mathbf{w}) = \frac12\sum_{n=1}^N [y_n-f_\mathbf{w}(\mathbf{x}_n)]^2+\frac\lambda2||\mathbf{w}||^2$

- 最速梯度下降
  - $\mathbf{w}^{\tau+1} = \mathbf{w}^{\tau} - \eta\nabla_\mathbf{w} J(\mathbf{w}^{\tau})$
    - 学习率参数： $\eta\in\mathcal{R}$
    - 与 一元回归 形式相同
  
- 停止条件
  - 当迭代次数 $\tau$  大于 某个值
  - 当 $|\eta\nabla_\mathbf{w} J(\mathbf{w}^{\tau})|$  小于 某个值

## 最小均方误差 / 随机梯度下降

- 代价函数
  - 平方和误差函数：
    - $J(\mathbf{w}) = \frac12 [y_n-f_\mathbf{w}(\mathbf{x}_n)]^2$
  - 正则化平方和误差函数：
    - $J(\mathbf{w}) = \frac12 [y_n-f_\mathbf{w}(\mathbf{x}_n)]^2+\frac\lambda2||\mathbf{w}||^2$

- 随机梯度下降
  - $\mathbf{w}^{\tau+1} = \mathbf{w}^{\tau} - \eta\nabla_\mathbf{w} J(\mathbf{w}^{\tau})$
    - 学习率参数： $\eta\in\mathcal{R}$
    - 与 一元回归 形式相同
  
- 停止条件
  - 当迭代次数 $\tau$  大于 某个值

## 梯度下降算法对比

- 最速梯度下降
  - 在每轮迭代中，选择全部数据优化损失函数
  - 矩阵计算成本高， 收敛速度快
  
- 随机梯度下降，也叫 序列梯度下降
  - 在每轮迭代中，顺序选择一条数据优化损失函数
  - 矩阵计算成本低， 收敛速度慢
  
- 批梯度下降
  - 在每轮迭代中，随机选择一批数据优化损失函数
  - 合理的矩阵计算成本，合理的收敛速度

::: notes

- 随机梯度下降：极小化代价函数的瞬时值
- 随机梯度下降收敛性：随机过程中的随机游走（布朗运动），收敛于渐近稳定的不动点

:::

## 概率建模：学习框架

- 线性有噪声模型
  - 无噪声函数：$f_\mathbf{w}(\dot{\mathbf{x}}) = \dot{\mathbf{x}}^T\mathbf{w}$
  - 概率建模：$y \sim \mathcal{N}(f_\mathbf{w}(\dot{\mathbf{x}}),\sigma)$  ， $y\in\mathcal{R}, \mathbf{w}\in\mathcal{R}^{D+1}, \dot{\mathbf{x}}\in\mathcal{R}^{D+1}$

- 代价函数
  - 离线学习
    - 最大似然估计
    - 最大后验估计
  - 在线学习：最小均方误差

## 最大似然估计

- 似然函数
  - $p(\mathbf{y}|\mathbf{x},\mathbf{w},\sigma)=\prod_{n=1}^N\mathcal{N}(y_n|f_{\mathbf{w}}(\mathbf{x}_n),\sigma^2)$
  - $\ln p(\mathbf{y}|\mathbf{x},\mathbf{w},\sigma) = -\frac{1}{2\sigma^2} \sum_{n=1}^N [y_n - f_{\mathbf{w}}(\mathbf{x}_n)]^2 + \frac{N}{2}\ln(\sigma^2)-\frac{N}{2}\ln(2\pi)$

- 解
  - $\mathbf{w}_{_{ML}}=\arg\min_{\mathbf{w}}\ln p(\mathbf{y}|\mathbf{x,w},\sigma)\propto\arg\min_{\mathbf{w}}\sum_{n=1}^N[y_n-f_{\mathbf{w}}(\mathbf{x}_n)]^2$
- 与 一元回归 形式相同
  
- 学习方法
  - 等价于  "平方误差函数" 的极值

## 最大后验估计

- 后验概率
  - $p(\mathbf{w}|\sigma_\mathbf{w}) = \mathcal{N}(\mathbf{w}|\boldsymbol{0},\sigma_w^2\mathbf{I})$
  - $p(\mathbf{w}|y,\mathbf{x},\sigma)\propto p(y|\mathbf{x},\mathbf{w},\sigma) p(\mathbf{w}|\sigma_\mathbf{w})$

- 解
  - $\mathbf{w}_{_{MAP}}=\arg\min_{\mathbf{w}} \ln p(\mathbf{w}|y,\mathbf{x},\sigma)\propto\arg\min_{\mathbf{w}}\sum_{n=1}^N[y_n-f_{\mathbf{w}}(x_n)]^2 + \frac{\lambda}{2}||\mathbf{w}||^2$
    - $\lambda=(\frac{\sigma}{\sigma_w})^2$
  - 与 一元回归 形式相同
  
- 学习方法
  - 等价于 “正则化平方误差函数” 的极值

# 线性基函数模型

## 一元回归：基函数建模

- 线性无噪声模型
  - $y = w_0 1 + w_1 x = \mathbf{w}^T\mathbf{x}$  , $x\in\mathcal{R}, y\in\mathcal{R}$

- 线性基函数无噪声模型
  - $y = w_0 g_0(x)+w_1 g_1(x)+\dots+w_p g_p(x) = \mathbf{w}^T\mathbf{g}(x)$  , $x\in\mathcal{R}, y\in\mathcal{R}$

- 线性基函数带噪声模型
  - 无噪声函数： $f(\mathbf{x}) = \mathbf{w}^T\mathbf{g}(x)$
  - 函数建模： $y =f(\mathbf{x}) + \varepsilon$  , $x\in\mathcal{R}, y\in\mathcal{R}, \varepsilon\sim\mathcal{N}(0,1)$
  - 概率建模： $y \sim \mathcal{N}(f(\mathbf{x}),1)$

# 广义线性模型

## 一元回归：广义线性模型

- 线性无噪声模型
  - $y = w_0 1 + w_1 x = \mathbf{w}^T\mathbf{x}$  , $x\in\mathcal{R}, y\in\mathcal{R}$

- 广义线性模型
  - $y = g(\mathbf{w}^T\mathbf{x})$  , $x\in\mathcal{R}, y\in\mathcal{R}$
