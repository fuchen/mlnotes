---
title: 回归与分类
date: 2017-04-18 23:34:14
category: 机器学习笔记
tags: [softmax, 逻辑回归, 线性回归]
mathjax2: true
---
### 线性回归(Linear Regression)

估值函数(Hypothesis Function):
`$$ h_\theta(x) = \theta x $$`

目标函数:
`$$ J(\theta) = \frac{1}{m}\sum_{i=1}^{m}Cost(h_\theta(x^{(i)}), y^{(i)}) $$`

`$$ Cost(h_\theta(x), y) = \frac{1}{2}(h_\theta(x) - y)^2 $$`

<!-- more -->

对 `$\theta$` 求一阶导数:
`$$ Cost'(\theta) = (h_x(\theta) - y) \cdot h_x'(\theta) = x^2\theta - xy $$`

对 `$\theta$` 求二阶导数:
`$$ Cost''(\theta) = x^2 > 0 $$`

可见 Cost Function 为凸函数，不会陷入到局部最优点。


### 逻辑回归(Logistic Regression)与分类

引入 Sigmoid(logistic) Function, 使 `$h_\theta(x)$` 归一化，即: `$ 0 < h_\theta(x) < 1 $`:
`$$ g(x) = \frac{1}{1 + e^{-x}} $$`

估值函数(Hypothesis Function):
`$$ h_\theta(x) = P(y=1 | x; \theta) = g(\theta x) = \frac{1}{1 + e^{-\theta x}} $$`

目标函数:
`$$ J(\theta) = \frac{1}{m}\sum_{i=1}^{m}Cost(h_\theta(x^{(i)}), y^{(i)}) $$`

由于估值函数不是线性的，如果采用线性回归的 Cost Function，那么会得到一条非凸曲线(nonconvex function)。为了使 `$ J(\theta) $`成为一个凸函数(convex function)，需要作出一些修改:

{% math %}
Cost(h_\theta(x), y) = \begin{cases}
-\log(h_\theta(x))     & \text{if y=1} \\
-\log(1 - h_\theta(x)) & \text{if y=0}
\end{cases}
{% endmath %}

`$$ Cost(h_\theta(x), y) = (y-1)\log(1 - h_\theta(x)) - y\log(h_\theta(x)) $$`


对 `$\theta$` 求一阶导数:
`$$ Cost'(\theta) = xe^{\theta x} + xe^{2\theta x} - xy $$`

对 `$\theta$` 求二阶导数:
`$$ Cost''(\theta) = x^2e^{\theta x} + 2x^2e^{2\theta x} > 0 $$`

所以 Cost Function 是凸函数，不会陷入到局部最优点。

### 基于 Logistic Regressioin 对多类别分类

针对每一类别，分别计算 `$ P(y=k | x; \theta) $`。可以取最大值，做__排他性分类__。也可以设定阀值，做__非排他性分类__。

### Softmax 分类

从 Logistic Regression 演变而来，用于做__排他性分类__。

{% math %}
h_\theta(x) =
\left[\begin{array}{c}
  P(y=1 | x; \theta) \\ P(y=2 | x; \theta) \\ \vdots \\ P(y=K | x; \theta)
\end{array}\right] =
\frac{1}{\sum_{j=1}^{K}e^{\theta^{(j)} x}} =
\left[\begin{array}{c}
  e^{\theta^{(1)} x} \\ e^{\theta^{(2)} x} \\ \vdots \\ e^{\theta^{(K)} x}
\end{array}\right]
{% endmath %}

{% math %}
\theta =
\left[\begin{array}{cccc}
  \mid & \mid & \mid & \mid \\
  \theta^{(1)} & \theta^{(2)} & \dots & \theta^{(K)} \\
  \mid & \mid & \mid & \mid \\
\end{array}\right]
{% endmath %}

目标函数:
{% math %}
J(\theta) = -
\left[
  \sum_{i=1}^{m}(1-y^{(i)})\log(1 - h_\theta(x^{(i)})) + y^{(i)}\log h_\theta(x^{(i)})
\right]
{% endmath %}

当 `$K=2$` 时，softmax 会退化成 logistic regression 的形式:
{% math %}
h_\theta(x) =
\left[\begin{array}{c}
  \frac{1}{1+e^{(\theta^{(1)} - \theta^{(2)})x }} \\
  1 - \frac{1}{1+e^{(\theta^{(1)} - \theta^{(2)})x }}
\end{array}\right]
{% endmath %}
