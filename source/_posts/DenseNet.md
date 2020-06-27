---
title: DenseNet
date: 2020-06-26 14:07:43
tags: 论文笔记
---



# DenseNet笔记
- [DenseNet笔记](#densenet笔记)
  - [1.摘要](#1摘要)
  - [2.网络结构](#2网络结构)
  - [3.结论](#3结论)
  - [代码](#代码)
    - [DenseBlock](#denseblock)
    - [Transition](#transition)
    - [DenseNet](#densenet)
## 1.摘要
在卷积网络中，如果靠近输出的层和靠近输入层有shorter connect的时候，网络可以设计的更深、具有更高的准确率以及更高效的训练。在这篇论文中，我们提出了DenseNet的结构，在DenseNet中每个层以前向的方式与后面的层进行连接。传统的具有L层的神经网络，具有L个连接，但是DenseNet中L层的网络具有$L(L+1)/2$这么多个连接。将每个卷积层网络的输入变为前面所有网络的输出的拼接。

<!--more-->

DenseNet网络具有以下几个优势：  
**1. 网络缓和了梯度消失**  
**2. 强化了网络中特征的前向传播**  
**3. 鼓励网络中特征的重用**  
**4. 减少了网络的参数**  

## 2.网络结构

<center>
 
 ![网络结构](http://paper.lmunan.online/20200625091448.png)
</center>

相比ResNet，DenseNet提出了一个更激进的密集连接机制：即互相连接所有的层，具体来说就是每个层都会接受其前面所有层作为其额外的输入。图1为ResNet网络的连接机制，作为对比，图2为DenseNet的密集连接机制。可以看到，ResNet是每个层与前面的某层（一般是2~3层）短路连接在一起，连接方式是通过元素级相加。而在DenseNet中，每个层都会与前面所有层在channel维度上连接（concat）在一起，并作为下一层的输入。 简单点说，**ResNet是直接相加，DenseNet是堆叠**。  

<center>

![ ResNet网络的短路连接机制](http://paper.lmunan.online/20200625092037.png)
ResNet网络的短路连接机制
</center>

<center>

![DenseNet网络的密集连接机制](http://paper.lmunan.online/20200625092134.png)
DenseNet网络的密集连接机制
</center>

如果用公式表示的话，传统的网络在L层的输出为：
$$ x_{l}=H_{l}(x_{l-1}^{}) $$

而对于ResNet，增加了来自上一层输入的identity函数：
$$ x_{l}=H_{l}(x_{l-1}^{})+x_{l-1}^{}$$

在DenseNet中，会连接前面所有层作为输入：
$$ x_{l}=H_{l}([x_{0},x_{1},\cdots ,x_{l-1}])$$

其中，上面的 $H_{l}(\cdot )$代表是非线性转化函数（non-liear transformation），它是一个组合操作，其可能包括一系列的BN(Batch Normalization)，ReLU，Pooling及Conv操作。注意这里 $l$层与 $l-1$层之间可能实际上包含多个卷积层。

<center>

![20200625093327](http://paper.lmunan.online/20200625093327.png)
一个完整的DenseNet网络结构
</center>

在DenseBlock中，各个层的特征图大小一致，可以在channel维度上连接。DenseBlock中的非线性组合函数 $H_{l}(\cdot )$采用的是BN+ReLU+3x3 Conv的结构。
由于后面层的输入会非常大，DenseBlock内部可以采用bottleneck层来减少计算量，主要是原有的结构中增加1x1 Conv，如图7所示，即**BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv**，称为DenseNet-B结构。

ResNet不同，所有DenseBlock中各个层卷积之后均输出 $k$ 个特征图，即得到的特征图的channel数为 $k$ ，或者说采用 $k$ 个卷积核。 $k$ 在DenseNet称为growth rate，这是一个超参数。一般情况下使用较小的 $k$ （比如12），就可以得到较佳的性能。假定输入层的特征图的channel数为 $k_{0}$ ，那么 $l$ 层输入的channel数为$k_{0}+k(l-1)$，因此随着层数增加，尽管$k$设定得较小，DenseBlock的输入会非常多，不过这是由于特征重用所造成的，每个层仅有$k$个特征是自己独有的。

对于Transition层，它主要是连接两个相邻的DenseBlock，并且降低特征图大小。Transition层包括一个1x1的卷积和2x2的AvgPooling，结构为BN+ReLU+1x1 Conv+2x2 AvgPooling。另外，Transition层可以起到压缩模型的作用。假定Transition的上接DenseBlock得到的特征图channels数为 $m$ ，Transition层可以产生 $\left \lfloor \theta _{m} \right \rfloor$个特征（通过卷积层），其中 $\theta \in (0,1]$ 是压缩系数（compression rate）。当 $\theta =1$ 时，特征个数经过Transition层没有变化，即无压缩，而当压缩系数小于1时，这种结构称为DenseNet-C，文中使用 $\theta =0.5$ 。对于使用bottleneck层的DenseBlock结构和压缩系数小于1的Transition组合结构称为DenseNet-BC。

<center>

![20200625095308](http://paper.lmunan.online/20200625095308.png)
</center>

## 3.结论
综合来看，DenseNet的优势主要体现在以下几个方面：

* 由于密集连接方式，DenseNet提升了梯度的反向传播，使得网络更容易训练。由于每层可以直达最后的误差信号，实现了隐式的“deep supervision”；
* 参数更小且计算更高效，这有点违反直觉，由于DenseNet是通过concat特征来实现短路连接，实现了特征重用，并且采用较小的growth rate，每个层所独有的特征图是比较小的；
* 由于特征复用，最后的分类器使用了低级特征。

## 代码

### DenseBlock

 ``` python
 #卷积块：BN->ReLU->1x1Conv->BN->ReLU->3x3Conv
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        "num_input_features:输入特征图个数  64
        growth_rate:增长速率，第二个卷积层输出特征图 32
        grow_rate * bn_size:第一个卷积层输出特征图 32*4
        drop_rate:dropout失活率 0"
        
        super(_DenseLayer, self).__init__()
        self.add_module('norm.1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu.1', nn.ReLU(inplace=True)),
        self.add_module('conv.1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm.2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu.2', nn.ReLU(inplace=True)),
        self.add_module('conv.2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        # 最后将新生成的featrue map和输入的feature map在channel维度上concat起来
        # 1.不需要像ResNet一样将x进行变换使得channel数相等
        # 因为DenseNet conv2 3*3conv stride=1 不会改变Tensor的h,w，并且最后是channel维度上的堆叠而不是相加
        # 2.原文中提到的内存消耗多也是因为这步，在一个block中需要把所有layer生成的feature都保存下来
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        "num_layers:每个block内dense layer层数"
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            # k0+k(l-1)个特征图输入
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

 ``` 

### Transition
```python
class _Transition(nn.Sequential):#过渡层，将特征图个数减半
    def __init__(self, num_input_features, num_output_features):
    "num_input_features:输入特征图个数
     num_output_features:输出特征图个数，为num_input_features//2"
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_featuresnum_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
```

### DenseNet
最后实现的DenseNet就是交替连接DenseBlock和Transition（最后一个DenseBlock接池化层和softmax分类器）。
```python
class DenseNet(nn.Module):
    
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # 第一个卷积层
        # 和ResNet一样，先通过7*7的卷积，将分辨率从224*224->112*112
        # 通过3*3最大池化，将分辨率从112*112->56*56,此时的图为(56,56,64)
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # 每个denseblock
        num_features = num_init_features
        # 读取每个Dense block层数的设定
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,num_input_features=num_features, bn_size=bn_size,growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            # 每两个dense block之间增加一个过渡层
            # 第四个Dense block后不再连接Transition层
            if i != len(block_config) - 1: 
                # 此处可以看到，默认过渡层将channel变为原来输入的一半
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        #  batch norm
        # 不知道为什么单独把BN写在这里，而把relu,avg_pool写到forward里。
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        # self.features.add_module('relu5', nn.ReLU(inplace=True))
        # self.features.add_module('avgpool5', nn.AvgPool2d(kernel_size=7, stride=1))


        # 分类器
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = self.classifier(out)
        return out
```

参考文章：

[DenseNet：比ResNet更优的CNN模型](https://zhuanlan.zhihu.com/p/37189203 "DenseNet：比ResNet更优的CNN模型")

[DenseNet论文翻译及pytorch实现解析（上）](https://zhuanlan.zhihu.com/p/31647627 "DenseNet论文翻译及pytorch实现解析（上）")

[DenseNet论文翻译及pytorch实现解析（下）](https://zhuanlan.zhihu.com/p/31650605 "DenseNet论文翻译及pytorch实现解析（下）")
