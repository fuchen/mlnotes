---
title: 神经网络
date: 2017-04-20 20:02:46
category: 机器学习
tags: ["神经网络", "反向传播"]
---
下图是一个4层的神经网络:
{% asset_img "Neural_Network.svg" "一个4层神经网络（2个隐层）"%}

假设 `$l$` 层有 `$m$` 个节点作出输入单元，`$l+1$` 层有 `$n$` 个节点作为输出单元，则:
{% math %}
w^{(l)} = \left[
  \begin{array}{c|cccc}
    w_{10}^{(l)} & w_{11}^{(l)} & w_{12}^{(l)} & \dots & w_{1m}^{(l)} \\
    w_{20}^{(l)} & w_{21}^{(l)} & w_{22}^{(l)} & \dots & w_{2m}^{(l)} \\
    \vdots & \vdots & \dots & \vdots & \vdots \\
    w_{n0}^{(l)} & w_{n1}^{(l)} & w_{n2}^{(l)} & \dots & w_{nm}^{(l)}
  \end{array}
\right]
{% endmath %}
其中，第一列对应的是偏置单元。

**引入bias的原因:** bias 和 weigth 是不同的加权方式，合起来才获得完全的线性能力，即: `$ y=wx +b $`

### 前向传播过程(Forward propagation):

令激活函数为 `$g$` :

{% math %}
\begin{align}
  &z^{(1)} = w^{(1)} x       & a^{(2)} = g(z^{(1)}) \\[10pt]
  &z^{(2)} = w^{(2)} a^{(2)} & a^{(3)} = g(z^{(2)}) \\[10pt]
  &z^{(3)} = w^{(3)} a^{(3)} & h(w, x) = a^{(4)} = g(z^{(3)})
\end{align}
{% endmath %}



### 反向传播过程(Back propagation):

首先计算最后一层的误差:
`$$ \delta^{(4)} = y - h(w, x) $$`

误差的反向传播，从 `$l+1$` 层(`$n$`个节点)到 `$l$` 层(`$m$`个节点):
{% math %}
\begin{align}
  & a_i^{(l+1)} = g(z_i^{(l)}) = g(\sum_{j=0}^{m} w_{ij}^{(l)} a_j^{(l)}) \\[10pt]
  \Rightarrow \,\,\, & \frac{\partial a_i^{(l+1)}}{\partial a_j^{(l)}} = w_{ij}^{(l)}g'(z_i^{(l)}) \\[10pt]
  \Rightarrow \,\,\, & \delta_j^{(l)} = \sum_{i=1}^{n} w_{ij}^{(l)}g'(z_i^{(l)}) \delta_i^{(l+1)} \\[10pt]
  \Rightarrow \,\,\, & \delta^{(l)} = w^{(l)T}(g'(z^{(l)}) \cdot \delta^{(l+1)})
\end{align}
{% endmath %}

利用第 `$l+1$` 层的误差来计算第 `$l$` 层的权值梯度，基本原理如下:
{% math %}
\begin{align}
  & a_i^{(l+1)} = g(z_i^{(l)}) = g(\sum_{j=0}^{m} w_{ij}^{(l)} a_j^{(l)}) \\[10pt]
  \Rightarrow \,\,\, & \frac{\partial a_i^{(l+1)}}{\partial w_{ij}^{(l)}} = a_j^{(l)} g'(z_i^{(l)}) \\[10pt]
  \Rightarrow \,\,\, & \Delta_{ij}^{(l)} =  a_j^{(l)} g'(z_i^{(l)}) \delta_i^{(l+1)} \\[10pt]
  \Rightarrow \,\,\, & \Delta^{(l)} = (g'(z^{(l)}) \cdot \delta^{(l+1)}) \times a^{(l)T}
\end{align}
{% endmath %}

然后计算一批训练数据的 `$\Delta$` 平均值，并以此来更新 `$w$`。

### Back Propagation 与 Regression 在形式上的区别:

* Regression 过程中，输入 `$x$` 是固定的，每次迭代只需要更新权值，即 `$\theta$` 或者说 `$w$`。
* Back Propagation 的第一层在形式上与 Regression 很相似，但在中间层上，每次迭代不仅更新了 `$w^{(l)}$`，而且更新了 `$a^{(l)}$`。
