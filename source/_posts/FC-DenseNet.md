---
title: FC-DenseNet
date: 2020-06-27 19:55:19
tags: 论文笔记
aplyer: true
---

# FC-DenseNet笔记
- [FC-DenseNet笔记](#fc-densenet笔记)
  - [摘要](#摘要)
  - [Dense block的结构如下图所示：](#dense-block的结构如下图所示)
  - [FC-DenseNet结构](#fc-densenet结构)
  - [FC-DenseNet103结构](#fc-densenet103结构)
## 摘要

目前最先进的语义图像分割方法是建立在卷积神经网（CNNs）上。
典型的语义分割网络结构组成如下：
1. 一条下采样路径，提取粗略的分割特征。
2. 一条上采样路径，恢复输入图像的分辨率。
3. 一个后处理模块，如条件随机场来对模型预测进行精修(可选)。

最近，一种新的CNN架构--密集连接卷积网络[DenseNet](https://www.lmunan.online/2020/06/26/DenseNet/)在图像分类任务上表现出了良好的效果。DenseNets 的想法是基于这样一种观察，即如果每一层以前馈的方式直接连接到其他每一层，都么网络将会更准确，更容易训练。

<!---more--->

**我们仅仅在dense模块后增加上采样通道，这使得每种分辨率的dense模块上采样通道与池化层个数无关，通过下采样和上采样间的跨层连接，高分辨率的信息得以传递。**

主要贡献：

1. 我们小心地将DenseNet扩展为用于语义分割的全卷积网络，同时缓解了特征图爆炸（feature map explosion）问题。
2. 我们提出的使用dense blocks构建的上采样路径，性能比标准的上采样路径更好，例如U-Net。
3. 该网络可以在城市街景数据集上取得最先进的结果，不需要任何预训练的参数或其他后处理步骤。

## Dense block的结构如下图所示：
<center>

![Dense block](http://paper.lmunan.online/20200628150207.png)
</center>

## FC-DenseNet结构

DenseNet结构组成了我们的全卷积DenseNet（FC-DenseNet）的下采样路径。注意在下采样路径中，特征图数量的线性增长通过池化操作降低特征图空间分辨率来补偿。下采样路径的最后一层被称为瓶颈（bottleneck）。

为了恢复空间分辨率，FCN提出使用卷积和上采样操作（转置卷积或去池化操作），以及skip connections组成的上采样路径。在FC-DenseNet中，我们将卷积操作替换为一个dense block和一个称为transition up的上采样操作。Transition up模块包含一个转置卷积来上采样前面的特征图。上采样后的特征图与来自skip-connection的特征图连接，组成下一个dense block的输入。

由于上采样路径提高了特征图的空间分辨率，特征图数量的线性增长会造成巨大的内存开销。为了缓解这个问题，上采样路径中，一个dense block的输入和输出不会被连接到一起。也就是说，转置卷积只对最后一个dense block的输出特征图进行操作，而不是把之前的所有连接在一起的特征图。

总体结构如图所示：

<center>

![20200628092241](http://paper.lmunan.online/20200628092241.png)

</center>



下图分别定义了dense block中的layer、transition down和transition up：
<center>

![20200628150549](http://paper.lmunan.online/20200628150549.png)
</center>

## FC-DenseNet103结构

第1层是输入，下采样路径共38层，瓶颈共15层，上采样路径共38层，5个transition down每个包含一个卷积，5个transitiona up每个包含一个转置卷积，最后1层是一个1×1卷积紧跟着softmax激活函数。
<center>

![20200628150701](http://paper.lmunan.online/20200628150701.png)
</center>