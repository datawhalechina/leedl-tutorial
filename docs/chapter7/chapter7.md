
![](BackPropagation.png)
[TOC]



# 文章说明

反向传播（Backpropagation）算法是怎么让神经网络（neural network）变的有效率的。

# 背景
## 梯度下降（Gradient Descent）
![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-1.png)
- 给到 $\theta$ (weight and bias)
- 先选择一个初始的 $\theta^0$，计算 $\theta^0$ 的损失函数（Loss Function）设一个参数的偏微分
- 计算完这个向量（vector）偏微分，然后就可以去更新的你 $\theta$ 
- 百万级别的参数（millions of parameters）
- 反向传播（Backpropagation）是一个比较有效率的算法，让你计算梯度（Gradient） 的向量（Vector）时，可以有效率的计算出来

## 链式法则（Chain Rule）
![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-2.png)
- 连锁影响
- BP主要用到了chain rule


# 反向传播

1. **损失函数(Loss function)是定义在单个训练样本上的**，也就是就算一个样本的误差，比如我们想要分类，就是预测的类别和实际类别的区别，是一个样本的哦，用L表示。
2. **代价函数(Cost function)是定义在整个训练集上面的**，也就是所有样本的误差的总和的平均，也就是损失函数的总和的平均，有没有这个平均其实不会影响最后的参数的求解结果。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-3.png)

那怎么去做呢？我们先整个神经网络（Neural network）中抽取出一小部分的神经（Neural）去看：

![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-4.png)
- cross entropy 到total loss 
- 问题是咋样计算每一笔data的partial

从这一小部分中去看，把计算梯度分成两个部分
- 计算$\frac{\partial z}{\partial w}$（Forward pass的部分）
- 计算$\frac{\partial C}{\partial z}$ ( Backward pass的部分 )


## 取出一个Neural进行分析
![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-5.png)

## Forward Pass

那么，首先计算$\frac{\partial z}{\partial w}$（Forward pass的部分）：
![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-6.png)

根据链式法则（Chain Rule），forward pass的运算规律就是：

$$\frac{\partial z}{\partial w_1} = x_1 \\ \frac{\partial z}{\partial w_2} = x_2$$

直接使用数字，更直观地看到运算规律：
![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-7.png)
是不是只管看到了结果，就是上一个的值，接下来是计算 Backward pass~~


## Backward Pass
 (Backward pass的部分)这就很困难复杂因为我们的C是最后一层：
那怎么计算 $\frac{\partial C}{\partial z}$ （Backward pass的部分）这就很困难复杂因为我们的C是最后一层：

![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-8.png)

计算所有激活函数的偏微分，激活函数有很多，这里使用Sigmod函数为例
![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-9.png)
这里使用链式法则（Chain Rule）的case1，计算过程如下：

$$\frac{\partial C}{\partial z} = \frac{\partial a}{\partial z}\frac{\partial C}{\partial a} \Rightarrow   {\sigma}'(z)$$ 

![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-10.png)
最终的式子结果：

![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-11.png)

但是你可以想象从另外一个角度看这个事情，现在有另外一个neural，把back的过程逆向过来：

![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-12.png)

### case 1 : Output layer
case 1，就是y1与y2是输出值：
![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-13.png)


### case 2 : Not Output Layer

![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-14.png)
- Compute $\frac{\partial l}{\partial z}$ for all activation function inputs z
- Compute $\frac{\partial l}{\partial z}$ from the output layer

![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-15.png)

怎么去计算呢？

![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-16.png)
实际上进行backward pass时候就是反向的计算(也相当于一个neural network)。

# 总结

![image](http://ppryt2uuf.bkt.clouddn.com/chapter7-17.png)


