@[toc]

> 本文主要学习深度学习的一些基础知识，了解入门
> 
# 背景
## 深度学习的发展趋势
下图是Google使用深度学习的项目变化趋势：
![image](https://img-blog.csdnimg.cn/20190402185926668.png)

## 深度学习的发展史
回顾一下deep learning的历史：
- 1958: Perceptron (linear model)
- 1969: Perceptron has limitation
- 1980s: Multi-layer perceptron 
	- Do not have significant difference from DNN today
- 1986: Backpropagation
	- Usually more than 3 hidden layers is not helpful
- 1989: 1 hidden layer is “good enough”, why deep?
- 2006: RBM initialization (breakthrough) 
- 2009: GPU
- 2011: Start to be popular in speech recognition
- 2012: win ILSVRC image competition 

感知机（Perceptron）非常像我们的逻辑回归（Logistics Regression）只不过是没有`sigmoid`激活函数。09年的GPU的发展是很关键的，使用GPU矩阵运算节省了很多的时间。

# 深度学习的三个步骤
我们都知道机器学习有三个step，那么对于deep learning呢？其实也是3个步骤~~如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190402190511925.png)
- Step1：神经网络（Neural network）
- Step2：模型评估（Goodness of function）
- Step3：选择最优函数（Pick best function）

那对于深度学习的`Step1`就是神经网络（Neural Network）

## Step1：神经网络（Neural Network）
神经网络（Neural network）里面的节点，类似我们的神经元。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190402192613421.png)
神经网络也可以有很多不同的连接方式，这样就会产生不同的结构（structure）。
那都有什么连接方式呢？其实连接方式都是你手动去设计的：

### 完全连接前馈神经网络
概念：前馈（feedforward）也可以称为前向，从信号流向来理解就是输入信号进入网络后，信号流动是单向的，即信号从前一层流向后一层，一直到输出层，其中任意两层之间的连接并没有反馈（feedback），亦即信号没有从后一层又返回到前一层。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190402192332301.png)

- 当已知权重和阈值时输入$(1,-1)$的结果
- 当已知权重和阈值时输入$(0,0)$的结果

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190402194630313.png)
所以可以把整个神经网络（neural network）看成是一个函数（function），如果神经网络中的权重和阈值都知道的话，就是一个已知的函数（也就是说，如果我们把参数都设置上去，这个神经网络其实就是一个函数）。他的输入是一个向量，对应的输出也是一个向量。

如果只是定义了一个神经网络的结构，但是不知道权重还有阈值怎么办呢？
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190402195340322.png)
给了结构就相当于定义了一个函数集（function set）

不论是做回归模型（linear model）还是逻辑回归（logistics regression）都是定义了一个函数集（function set）。我们可以给上面的结构的参数设置为不同的数，就是不同的函数（function）。这些可能的函数（function）结合起来就是一个函数集（function set）。这个时候你的函数集（function set）是比较大的，是以前的回归模型（linear model）等没有办法包含的函数（function），所以说深度学习（Deep Learning）能表达出以前所不能表达的情况。

我们通过另一种方式显示这个函数集：

#### 全链接和前馈的理解
- 输入层（Input Layer）：1层
- 隐藏层（Hidden Layer）：N层
- 输出层（Output Layer）：1层
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190402195954613.png)
- 为什么叫全链接呢？
	- 因为layer1与layer2之间两两都有连接，所以叫做Fully Connect；
- 为什么叫前馈呢？
	- 因为现在传递的方向是由后往前传，所以叫做Feedforward。

#### 深度的理解
那什么叫做Deep呢？Deep = Many hidden layer。那到底可以有几层呢？
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190402212023389.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190402211718128.png)
- 2012 AlexNet：8层
- 2014 VGG：19层
- 2014 GoogleNet：22层
- 2015 Residual Net：152层
- 101 Taipei：101层

随着层数变多，错误率降低，随之运算量增大，通常都是超过亿万级的计算。对于这样复杂的结构，我们一定不会一个一个的计算，对于亿万级的计算，使用loop循环效率很低。

引入矩阵计算（Matrix Operation）能使得我们的运算的速度以及效率高很多：

### 矩阵计算
如下图所示，输入是 （1，-1），输出是（0.98，0.12）。
计算方法就是：sigmod（权重w【黄色】 * 输入【蓝色】+ 偏移量b【绿色】）= 输出
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190402212333192.png)
如果有很多层呢？
$$a^1 = \sigma (w^1x+b^1) \\
a^2 = \sigma (w^1a^1+b^2) \\ 
··· \\ 
y = \sigma (w^La^{L-1}+b^L) $$

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190402212747341.png)

计算方法就像是嵌套，这里就不列公式了，结合上一个图更好理解。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190402213705989.png)
从结构上看每一层的计算都是一样的，在计算机里面使用并行计算技术加速矩阵运算。
这样写成矩阵运算的好处是，你可以使用GPU加速。那我们看看本质是怎么回事呢？

#### 本质：通过隐藏层进行特征转换
疑问：PPT里面特征提取替代特征工程（Feature extractor replacing feature engineering），这句话没有看明白
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190402213943963.png)
#### 示例：手写数字识别
举一个手写数字体识别的例子
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190402214725116.png)
输入：一个16*16=256个特征的向量，有颜色用（ink）用1表示，没有颜色（no ink）用0表示
输出：10个维度，每个维度代表一个数字的置信度。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190402214422652.png)
从输出结果看，是数字2的置信度为0.7，比较高。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190402214901839.png)
将识别手写数字的问题转换，输入是256维的向量，输出是10维的向量，我们所需要求的就是隐藏层神经网络的函数
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019040221520171.png)
神经网络的结构，决定了函数集（function set），所以说网络结构（network structured）很关键。

了解Step1，也会引入相关的问题：
- 多少层？ 每层有多少神经元？
	- 尝试发现错误 + 直觉 
- 结构可以自动确定吗？
	-  进化人工神经网络（Evolutionary Artificial Neural Networks）
- 我们可以设计网络结构吗？
	- CNN卷积神经网络（Convolutional Neural Network ）
![image](http://upload-images.jianshu.io/upload_images/3982944-561db80b8bfb53b4.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
讲完了step1之后就要将step2(怎么定义参数的好坏)：

## Step2: 模型评估（Goodness of function）
![image](http://upload-images.jianshu.io/upload_images/3982944-08e3b1dbdc98c25c.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 损失示例

![image](http://upload-images.jianshu.io/upload_images/3982944-1b3ec35c9809d622.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们需要把所有的data算起来：
### 总体损失
![image](http://upload-images.jianshu.io/upload_images/3982944-205153c02c8d6d4f.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

<figcaption style="margin-top: 0.66667em; padding: 0px 1em; font-size: 0.9em; line-height: 1.5; text-align: center; color: rgb(153, 153, 153);">total loss</figcaption>

## Step3：选择最优函数（Pick best function）

### 梯度下降（Gradient Descent）
如何找呢？Gradient Descent方法：

![image](http://upload-images.jianshu.io/upload_images/3982944-d40c54155b91aa2a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

现在有很多架构工具了：
### 反向传播
![image](http://upload-images.jianshu.io/upload_images/3982944-3a999984dd440373.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)



# 思考
为什么要用深度学习，深层架构带来哪些好处？那是不是隐藏层越多越好？

## 隐藏层越多越好？
![image](http://upload-images.jianshu.io/upload_images/3982944-feb65b52fffdea20.jpg)
从图中展示的结果看，毫无疑问，层次越深效果越好~~

## 普遍性定理
参数多的model拟合数据很好是很正常的。下面有一个通用的理论：
为什么“深层”神经网络不是“胖”神经网络？可以通过具有一个隐藏层的网络实现（给定足够的隐藏神经元）
![image](http://upload-images.jianshu.io/upload_images/3982944-4f12f6dc90e9fb01.jpg)

# 参考文档
[神经网络和深度学习之——前馈神经网络](https://www.cnblogs.com/vipyoumay/p/9322316.html)
