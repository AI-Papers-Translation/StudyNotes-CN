# Transformer 特征提取器

<img src="pictures/image-20200901104633208.png" alt="image-20200901104633208" style="zoom:50%;" />

(图1-Transformer模型结构)

课件地址：
链接：http://47.93.208.249:9825/tree/0.Teacher/Online
密码：807d4a2c

## 模型序列

模型的输入

-   输入序列：$\text{inputs}=(i_1,i_2,\cdots,i_p,\cdots,i_N)$，其中$i_p\in\mathbb{N}^*$为输入符号表中的序号。用于图1中的$\text{Inputs}$

-   目标序列：$\text{targets}=(t_1,t_2,\cdots,t_q,\cdots,t_M)$，其中$t_q\in\mathbb{N^*}$为目标符号表中的序号。用于图1中的$\text{Outputs}$

模型的输出

-   输出序列：$\text{outputs}=(o_1,o_2,\cdots,o_q,\cdots,o_M)$，其中$o_q\in\mathbb{N}^*$为目标符号表中的序号。用于图1中的$\text{Output Probabilities}$

$$
\begin{aligned}
	\text{outputs}=Transformer\text{(inputs,targets)}\\
	\text{loss function}=\mathcal{L}(\text{targets,outputs})
\end{aligned}
$$

## 序列编码与位置编码

### 序列编码

输入序列的词嵌入编码：$Embedding(\text{inputs})\in\mathbb{R}^{N\times d_{\text{model}}}$

-   $N$为输入序列的长度
-   $d_{\text{inputs}}$为输入序列词嵌入的维度

目标序列的词嵌入编码：$Embedding(\text{targets}\in\mathbb{R}^{M\times d_{\text{model}}})$

-   $M$为目标序列的长度
-   $d_{\text{outputs}}$为输出序列词嵌入的维度

### 位置编码

编码函数：

$$
\begin{aligned}
    Pos\_Enc(\text{pos},2i)=\sin(\text{pos}/10000^{2i/d_{\text{model}}})\\
    Pos\_Enc(\text{pos},2i+1)=\cos(\text{pos}/10000^{2i/d_{\text{model}}})
\end{aligned}
$$

注：使用 sin 和 cos 函数，是因为基于和差化积公式，<!--TODO：公式推导-->

输入序列的位置编码：$Pos\_Enc(\text{inputs_position})\in\mathbb{R}^{N\times d_{\text{model}}}$

-   $\text{inputs_position}\in\{1,2,\cdots,p,\cdots,N\}$为输入序列中输入符号对应的位置序列
    -   $\text{pos}\in\text{inputs_position}$
    -   $i\in\{1,2,\cdots,d_{\text{model}}/2\}$

目标序列的位置编码：$Pos\_End(\text{targets_position})\in\mathbb{R}^{N\times d_{\text{model}}}$

-   $\text{targets_position}=\{1,2,\cdots,q,\cdots,M\}$为目标序列中输入符号对应的位置序列
    -   $\text{pos}\in\text{targets_position}$
    -   $i\in\{1,2,\cdots,d_{\text{model}}/2\}$

## Encoder

### Encoder Structure

编码器的结构

-   输入层：$e_0=Embedding(\text{inputs})+Pos\_Enc(\text{inputs_position})，e_0\in\mathbb{R}^{N\times d_{model}}$
-   编码层：$e_l=EncoderLayer(e_{l-1}),e_l\in\mathbb{R}^{N\times d_{model}},l\in[1,n]$
    -   编码层的输入：$e_{\text{in}}\in\mathbb{R}^{N\times d_{\text{model}}}$
    -   编码层的过渡：$e_{\text{mid}}=LayerNorm(e_{\text{in}}+MultiHeadAttention(e_{\text{in}}))$
        -   多头注意力机制：$MultiHeadAttention(\cdot)$
        -   前馈神经网络：$FFN(\cdot)$
        -   层归一化：$LayerNorm(\cdot)$
    -   编码层的输出：$e_{\text{out}}=LayerNorm(e_{\text{mid}}+FFN(e_{\text{mid}})),e_{\text{out}}\in\mathbb{R}^{N\times d_{\text{model}}}$
    -   编码器的编码层的叠加数目：$n$