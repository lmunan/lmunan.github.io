---
title: ResNet笔记
date: 2020-06-23 20:41:13
tags: 论文笔记
---


# ResNet论文笔记

随着CNN网络深度的增加，出现了两个主要问题：

+ 梯度消失或梯度爆炸
+ 退化问题
  
<!-- more -->
<center>

![_20200623103412](http://cdn.lmunan.online/_20200623103412.png)
</center>

上图显示了常规的CNN网络在训练集和测试集上都出现了随着网络层数的增加，误差反而增加的现象。ResNet提出一个残差学习框架去减轻训练深度网络的难度。


ResNet网络中的主要亮点：

+ 超深的网络结构（超过1000层）
+ 提出了residual模块
+ 使用了Batch Normalization加速训练（丢弃dropout）

## 残差学习
ResNet的主要思想是在网络中增加了直连通道，即Highway Network的思想。此前的网络结构是对输入做一个非线性变换，而Highway Network则允许保留之前网络层的一定比例的输出。ResNet的思想和Highway Network的思想也非常类似，允许原始输入信息直接传到后面的层中，如下图所示

<center>

![20200623103905](http://cdn.lmunan.online/20200623103905.png)
</center>  


## ResNet网络结构
<center>

![20190708154219999](http://cdn.lmunan.online/20190708154219999.png)
</center>

## 两种不同的残差单元
下面我们再分析一下残差单元，ResNet使用两种残差单元，如下图所示。左图对应的是浅层网络，而右图对应的是深层网络。对于短路连接，当输入和输出维度一致时，可以直接将输入加到输出上。但是当维度不一致时（对应的是维度增加一倍），这就不能直接相加。有两种策略：（1）采用zero-padding增加维度，此时一般要先做一个downsample，可以采用stride=2的pooling，这样不会增加参数；（2）采用新的映射（projection shortcut），一般采用1x1的卷积，这样会增加参数，也会增加计算量。短路连接除了直接使用恒等映射，当然都可以采用projection shortcut。

对于每个残差函数F，我们使用堆叠3层而不是2层。三层分别是1×1，3×3和1×1卷积，其中的两个1×1卷积层分别负责降低维度和增加（恢复）维度，从而在3×3卷积层这里产生一个瓶颈。

<center>

![20200623110206](http://cdn.lmunan.online/20200623110206.png)
![20200623110336](http://cdn.lmunan.online/20200623110336.png)
![20200623110401](http://cdn.lmunan.online/20200623110401.png)
</center>


### 参考视频 
[ResNet网络结构，BN以及迁移学习详解](https://www.bilibili.com/video/BV1T7411T7wa?t=1243 "ResNet")  
[使用pytorch搭建ResNet并基于迁移学习训练](https://www.bilibili.com/video/BV14E411H7Uw "ResNet")
