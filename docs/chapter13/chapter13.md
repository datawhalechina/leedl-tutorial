# Brief Introduction of Deep Learning
![](res/chapter13-0.png)
# 深度学习的发展趋势
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
我们都知道机器学习有三个step，对于deep learning其实也是3个步骤：

![](res/chapter13-1.png)
- Step1：神经网络（Neural network）
- Step2：模型评估（Goodness of function）
- Step3：选择最优函数（Pick best function）

那对于深度学习的Step1就是神经网络（Neural Network）

## Step1：神经网络（Neural Network）
神经网络（Neural network）里面的节点，类似我们的神经元。
![](res/chapter13-2.png)

神经网络也可以有很多不同的连接方式，这样就会产生不同的结构（structure）在这个神经网络里面，我们有很多逻辑回归函数，其中每个逻辑回归都有自己的权重和自己的偏差，这些权重和偏差就是参数。
那这些神经元都是通过什么方式连接的呢？其实连接方式都是你手动去设计的。

### 完全连接前馈神经网络（Full Connect Feedforward Network）
概念：前馈（feedforward）也可以称为前向，从信号流向来理解就是输入信号进入网络后，信号流动是单向的，即信号从前一层流向后一层，一直到输出层，其中任意两层之间的连接并没有反馈（feedback），亦即信号没有从后一层又返回到前一层。
![](res/chapter13-3.png)
- 当已知权重和偏差时输入$(1,-1)​$的结果
- 当已知权重和偏差时输入$(-1,0)$的结果
![](res/chapter13-4.png)

上图是输入为1和-1的时候经过一系列复杂的运算得到的结果
![](res/chapter13-5.png)

当输入0和0时，则得到0.51和0.85，所以一个神经网络如果权重和偏差都知道的话就可以看成一个函数，他的输入是一个向量，对应的输出也是一个向量。不论是做回归模型（linear model）还是逻辑回归（logistics regression）都是定义了一个函数集（function set）。我们可以给上面的结构的参数设置为不同的数，就是不同的函数（function）。这些可能的函数（function）结合起来就是一个函数集（function set）。这个时候你的函数集（function set）是比较大的，是以前的回归模型（linear model）等没有办法包含的函数（function），所以说深度学习（Deep Learning）能表达出以前所不能表达的情况。

我们通过另一种方式显示这个函数集：
#### 全链接和前馈的理解
- 输入层（Input Layer）：1层
- 隐藏层（Hidden Layer）：N层
- 输出层（Output Layer）：1层
![](res/chapter13-6.png)
- 为什么叫全链接呢？
	- 因为layer1与layer2之间两两都有连接，所以叫做Fully Connect；
- 为什么叫前馈呢？
	- 因为现在传递的方向是由后往前传，所以叫做Feedforward。
#### 深度的理解
那什么叫做Deep呢？Deep = Many hidden layer。那到底可以有几层呢？这个就很难说了，以下是老师举出的一些比较深的神经网络的例子
![](res/chapter13-7.png)
![](res/chapter13-8.png)
- 2012 AlexNet：8层
- 2014 VGG：19层
- 2014 GoogleNet：22层
- 2015 Residual Net：152层
- 101 Taipei：101层

随着层数变多，错误率降低，随之运算量增大，通常都是超过亿万级的计算。对于这样复杂的结构，我们一定不会一个一个的计算，对于亿万级的计算，使用loop循环效率很低。

这里我们就引入矩阵计算（Matrix Operation）能使得我们的运算的速度以及效率高很多：

### 矩阵计算
如下图所示，输入是 $$\begin{bmatrix}&1&-2\\ &-1&1\end{bmatrix}$$，输出是$$\begin{bmatrix}&0.98\\ &0.12\end{bmatrix}$$。
计算方法就是：sigmoid（权重w【黄色】 * 输入【蓝色】+ 偏移量b【绿色】）= 输出
![](res/chapter13-9.png)

其中sigmoid更一般的来说是激活函数(activation function)，现在已经很少用sigmoid来当做激活函数。

如果有很多层呢？
$$a^1 = \sigma (w^1x+b^1) \\
a^2 = \sigma (w^1a^1+b^2) \\ 
··· \\ 
y = \sigma (w^La^{L-1}+b^L) ​$$

![](res/chapter13-10.png)

计算方法就像是嵌套，这里就不列公式了，结合上一个图更好理解。所以整个神经网络运算就相当于一连串的矩阵运算。
![](res/chapter13-11.png)

从结构上看每一层的计算都是一样的，也就是用计算机进行并行矩阵运算。
这样写成矩阵运算的好处是，你可以使用GPU加速。
整个神经网络可以这样看：
### 本质：通过隐藏层进行特征转换
把隐藏层通过特征提取来替代原来的特征工程，这样在最后一个隐藏层输出的就是一组新的特征（相当于黑箱操作）而对于输出层，其实是把前面的隐藏层的输出当做输入（经过特征提取得到的一组最好的特征）然后通过一个多分类器（可以是softmax函数）得到最后的输出y。
![](res/chapter13-12.png)
### 示例：手写数字识别
举一个手写数字体识别的例子：
输入：一个16*16=256维的向量，每个pixel对应一个dimension，有颜色用（ink）用1表示，没有颜色（no ink）用0表示
输出：10个维度，每个维度代表一个数字的置信度。
![](res/chapter13-13.png)

从输出结果来看，每一个维度对应输出一个数字，是数字2的概率为0.7的概率最大。说明这张图片是2的可能性就是最大的
![](res/chapter13-14.png)

在这个问题中，唯一需要的就是一个函数，输入是256维的向量，输出是10维的向量，我们所需要求的函数就是神经网络这个函数

![](res/chapter13-15.png)
从上图看神经网络的结构决定了函数集（function set），所以说网络结构（network structured）很关键。

![](res/chapter13-16.png)

接下来有几个问题：
- 多少层？ 每层有多少神经元？
这个问我们需要用尝试加上直觉的方法来进行调试。对于有些机器学习相关的问题，我们一般用特征工程来提取特征，但是对于深度学习，我们只需要设计神经网络模型来进行就可以了。对于语音识别和影像识别，深度学习是个好的方法，因为特征工程提取特征并不容易。
- 结构可以自动确定吗？
有很多设计方法可以让机器自动找到神经网络的结构的，比如进化人工神经网络（Evolutionary Artificial Neural Networks）但是这些方法并不是很普及 。
- 我们可以设计网络结构吗？
可以的，比如 CNN卷积神经网络（Convolutional Neural Network ）

## Step2: 模型评估（Goodness of function）

### 损失示例

![](res/chapter13-17.png)

对于模型的评估，我们一般采用损失函数来反应模型的好差，所以对于神经网络来说，我们采用交叉熵（cross entropy）函数来对$y$和$\hat{y}​$的损失进行计算，接下来我们就是调整参数，让交叉熵越小越好。
### 总体损失
![](res/chapter13-18.png)

对于损失，我们不单单要计算一笔数据的，而是要计算整体所有训练数据的损失，然后把所有的训练数据的损失都加起来，得到一个总体损失L。接下来就是在function set里面找到一组函数能最小化这个总体损失L，或者是找一组神经网络的参数$\theta$，来最小化总体损失L

## Step3：选择最优函数（Pick best function）
如何找到最优的函数和最好的一组参数呢，我们用的就是梯度下降，这个在之前的视频中已经仔细讲过了，需要复习的小伙伴可以看前面的笔记。
### 梯度下降（Gradient Descent）
![](res/chapter13-19.png)
![](res/chapter13-20.png)

具体流程：$\theta$是一组包含权重和偏差的参数集合，随机找一个初试值，接下来计算一下每个参数对应偏微分，得到的一个偏微分的集合$\nabla{L}$就是梯度,有了这些偏微分，我们就可以不断更新梯度得到新的参数，这样不断反复进行，就能得到一组最好的参数使得损失函数的值最小

### 反向传播
![](res/chapter13-21.png)

在神经网络中计算损失最好的方法就是反向传播，我们可以用很多框架来进行计算损失，比如说TensorFlow，theano，Pytorch等等



# 思考
为什么要用深度学习，深层架构带来哪些好处？那是不是隐藏层越多越好？

## 隐藏层越多越好？
![](res/chapter13-22.png)

从图中展示的结果看，毫无疑问，层次越深效果越好~~

## 普遍性定理
![](res/chapter13-23.png)

参数多的model拟合数据很好是很正常的。下面有一个通用的理论：
对于任何一个连续的函数，都可以用足够多的隐藏层来表示。那为什么我们还需要‘深度’学习呢，直接用一层网络表示不就可以了？在接下来的课程我们会仔细讲到


