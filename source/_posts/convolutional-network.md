---
title: 卷积神经网络
date: 2017-04-24 21:25:28
category: 机器学习笔记
tags: [神经网络, 卷积神经网络, CNN]
description:
---

{% asset_image LeNet.jpg "LeNet" %}
<!-- more -->


### 卷积神经网络的构成:

- **输入层**：可以是任意维度的数据。一般要求每个维度的长度都满足 `$2^n$`，以便池化。
- **卷积层**：通过`K`个卷积核，把输入层转化为`K`个特征图。
- **池化层**：压缩特征图大小，以提高训练速度，和避免过拟合。
- **全连接层**：映射到结果集。

### 卷积层

{% asset_image conv_layer.png "卷积操作" %}

{% asset_image conv_kernel.png "卷积核 - 匹配失败" %}
{% asset_image conv_kernel_2.png "卷积核 - 匹配成功" %}

超参数：
- **`depth`**: 卷积核的数量
- **`size`**: 卷积核的大小。维度与输入数据相同。
- **`stride`**: 采样间距。由于存在池化/Dropout，`stride`通常设置为 `1`。
- **`padding`**: Padding规则。

一个卷积神经网络可能包含多个卷积层。

越靠前面的层，提取到的特征越简单，卷积核数量越少。越靠后面的层，卷积核数量越多。这是因为其输入已经不是像素点，而是具有一定复杂度的低级特征，组合爆炸可以构建出更多的复杂特征。

换个角度看，越前面的层，其卷积核的感受野越小，特征数也少。越后面的层，其卷积核的感受野越大，特征数自然也就更多。

### 经典的卷积神经网络结构

1. LeNet：其结构图见本文开头。每个卷积层对应一个池化层。
2. [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)：
  {% asset_image AlexNet.png "AlexNet" %}
  结构特点为：
  - 增加了网络深度
  - 使用了连续的卷积层

  其他：
  - 使用PCA方法给RGB加了噪声，以降低过拟合
3. ZF Net：结构同 AlexNet。区别在于每个卷积层的卷积核数量加倍，第一层的卷积核大小和stride缩小。
4. [GoogleNet (Inception-v4)](https://arxiv.org/pdf/1602.07261.pdf)：
  {% asset_image GoogleNet_Inception.png "Incepiton Module" %}
  主要特点：
  - 使用了多种不同的 Inception Module，部分取代单一路径的卷积层。Inception Module 里的多条路径，可以理解为多种不同的功能。反向传播时，自动选择最合适的功能进行加强。
  - 顶层用 Avarage Pooling 取代全连接。
  - 注意，Inception Module 里包含了 1X1 Conv。1X1 Conv 之所以有存在意义，是因为卷积操作是具有深度的。不同深度层次上的值，代表了不同的feature，所以 1X1 Conv 可以提取该点的 feature 组合。
5. [VGGNet](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)：从头到尾 3X3 Conv + 2X2 Pooling，但就是深。
6. [ResNet](https://arxiv.org/abs/1512.03385)：
  {% asset_image ResNet_shortcut.png "ResNet Shortcut" %}
  主要特点：
  - 每隔几层引入了一条 shortcut，把网络深度的增加变成了一个逐步集成的过程。即每加一层，性能不应该变得更差。由于加法操作的存在，反向传播的时候，图中的 X identity 能起到稳定器的作用。
  - 与 GoogleNet 一样，顶层也采用了 Avg Pool。
