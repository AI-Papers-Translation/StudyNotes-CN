# 机器学习知识体系

## 数学

### 数学基础

#### 微积分

##### 导数与积分

-   极限
-   函数的求导法则
-   积分的收敛性
-   微分方程

##### 极值

-   泰勒公式
-   求解函数的极值

##### 级数

-   级数的收敛性
-   幂级数 与 Fourier 级数
-   幂级数的线性无关性与 Fourier 级数的正交性

#### 线性代数

##### 矩阵

-   行列式
-   标量、向量、矩阵、张量
-   矩阵计算：基本计算、转置、求逆、内积

##### 线性变换

-   线性关系
-   正交关系
-   本征值 与 本征向量

##### 二次型

-   二次型的标准形 ( 法式 )
-   二次型的矩阵及其特征值
-   正定二次型

#### 概率论

##### 随机变量

-   离散 与 连续
-   值变量 与 向量变量
-   数字特征：均值 与 方差

##### 贝叶斯定理

-   先验概率
-   似然函数
-   后验概率

##### 极限定理

-   大数定律
-   中心极限定理
-   高斯分布

### 数学进阶

#### 泛函分析

##### 变分法

-   泛函极值

##### 赋范线性空间

-   数列空间
-   函数空间
-   对偶空间

##### 函数变换

-   函数卷积
-   Fourier 变换

#### 随机过程

##### 随机过程的概念

-   分布函数
-   数字特征

##### Markov 随机过程

-   无后效性
-   转移概率
-   极限概率

##### 平稳随机过程

-   时间平稳性
-   各态历经性
-   功率谱密度

#### 数理统计

##### 参数估计

-   点估计 与 区间估计
-   最大似然估计
-   最大后验估计

##### 随机模拟

-   抽样分布
-   Bootstrap
-   MCMC 抽样、Gibbs 抽样

##### 假设检验

-   参数假设检验
-   非参数假设检验

##### 线性模型

-   回归分析
-   方差分析

##### 多元数据分析

-   主成分分析
-   因子分析
-   聚类分析

### 应用数学

#### 最优化

##### 线性规划

-   线性规划问题：求线性目标函数在具有线性等式或者线性不等式的约束条件下的极值 ( 极大值或者极小值 ) 问题。
-   凸函数与凸规划
-   对偶理论：将一个标准形的线性规划问题，使用同样的参数另一个线性规划问题描述，称新问题为原问题的对偶问题

##### 数值最优化

-   无约束最优化的数值方法
    -   最速下降法
    -   牛顿法
    -   共轭梯度法

#### 数学建模

##### 数学模型

-   数学模型：是用数学符号对一类实际问题或者实际系统中发生的现象的 ( 近似 ) 描述
-   数学建模：获得这个模型，求解这个模型，并且得出结论以及验证结论是否正确的全过程。
-   建模三要素
    -   怎样从实际出发作出合理的假设，从而得到可以执行的数学模型
    -   怎样求解模型中出现的数学问题
    -   怎样验证模型的正确性和可行性

##### 建模方法论

-   问题分析
-   建立模型
-   数据处理
-   求解模型
-   模型验证

## 机器学习

### 机器学习基础

#### 回归问题

##### 线性回归模型

$$
y=f ( \text{x} ) +\epsilon=\text{w}^T\text{x}+\epsilon
$$

##### 线性基函数回归模型

$$
y=f ( \text{x} ) +\epsilon+\text{w}^T\boldsymbol{\phi ( \text{x} )}+\epsilon
$$

##### 非线性回归模型

$$
y=f ( \text{w}^T \text{x} ) +\epsilon
$$

#### 分类问题

##### 线性判别函数(连续特征)

-   二分类模型：$y ( \text{x} ) =\text{w}^T\text{x}$
    -   决策面：$D$ 维超平面
-   多分类算法
    -   「一对多」
    -   「一对一」
    -   缺点：产生无法分类的区域
-   参数学习
    -   基于最小平方方法
    -   基于线性判别分析
    -   基于感知机算法

##### Logistic 回归(连续特征)

-   二分类模型：$p(C_1|\text{x})=\sigma(\text{w}^T\text{x)}$
    -   $\sigma(\cdot)$ 是 Logistic Sigmoid 函数
    -   基于数值迭代方式求解
-   多分类模型：$p ( \mathcal{C}_k|\text{x} )) =\frac{\exp \{ \text{w}_k^T \text{x}  \}}{\sum_j\exp \{ \text{w}_j^T \text{x} \}}$
    -   基于交叉熵误差函数
    -   基于数值迭代方式求解

##### 朴素 Bayes 分类(离散特征)

$$
P ( \mathcal{C}_k |\text{x} ) =\frac{P ( \mathcal{C}_k ) P ( \text{x}|\mathcal{C}_k )}{P ( \text{x} )}=\frac{P ( c )}{P ( \text{x} )}\prod_{k=1}^K P ( \text{x}_i|\mathcal{C}_k )
$$

##### 概率生成式模型 与 概率判别式模型

#### 标注问题

### 机器学习进阶

#### 神经网络

#### 支持向量机

#### 深度学习

### 应用机器学习

#### 自然语言处理

#### 语音信号处理

#### 图像信号处理

## 计算机

### 计算机基础

#### 软件开发

##### 数据结构和算法

##### 开发语言与开发工具

##### 数据存储和数据库系统

#### 系统集成

##### 代码测试

-   白盒测试

-   黑盒测试

-   功能测试

##### 集成测试

##### 迭代开发

#### 项目实施

##### 实施准备

-   环境准备
    -   硬件：主机、客户端、并发架构
    -   软件：OS、VM、DLL
    -   网络：带宽、并发、安全
-   备份：硬件备份、数据备份、人员备份
-   技术准备
-   业务准备
-   安全保障

##### 系统上线

-   系统部署
-   应用部署
-   数据迁移
-   应急方案

##### 后期维护

### 计算机进阶

#### 版本管理

##### 里程碑

##### 版本升级

##### 版本分支

#### 项目管理

##### 需求管理

##### 进度管理

##### 人员管理

#### 质量管理

### 计算机项目

#### 电信

##### 通话记录

##### 消费记录

##### 投诉记录

#### 公安

##### 人口系统

##### 案件系统

#### 数据交换平台

##### 数据交换总线

##### 数据交换标准

##### 数据交换性能