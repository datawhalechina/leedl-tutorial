

[TOC]

# 概要
> 本文主要是李宏毅机器学习的笔记，主要介绍在训练DNN过程中，不同阶段用到的一些技巧，本文所用到的图示来自随堂PPT

## 深度学习的流程（Recipe of deep learning）
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-1.png)

三个步骤
- define a set function
- goodness of function
- pick the best function

做完这些事情后，你会得到一个neural network，你要检查的是，这个neural network在你的training set有没有得到好的结果，没有的话，回头看，是哪个步骤出了什么问题，你可以做什么样的修改，在training set得到好的结果

假如说你在training set得到了一个好的结果了，然后再把neural network放在你的testing data，testing set的performance才是我们关心的结果

如果在testing data performance不好，才是overfitting，(overfitting不是我们遇到的第一个问题)

如果training set上的结果变现不好，那么就要去neural network在一些调整，如果在testing set表现的很好，就意味成功了


# 不要总是抱怨过拟合（Do not always blame overfitting）

![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-2.png)

不要看到所有不好的performance都是overfitting

在testing data上看到一个56-layer和20-layer，显然20-layer的error较小，那么你就说是overfitting，那么这是错误的。首先你要检查你在training data上的结果(对于Neural network是需要检查这件事情)

在training data上56-layer的performance本来就比20-layer变现的要差很多，在做neural network时，有很多的问题使你的train不好，比如local mininmize等等，56-layer可能卡在一个local minimize上，得到一个不好的结果，这样看来，56-layer并不是overfitting，只是没有train的好


![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-3.png)
在deep learnin 文件上，当你读到一个方式的时候，你永远要想一下说，它是要解什么样的问题，在deep learning 中一个traininf data的performance不好，一个是testing data performace不好

当一个方法要被approaches时，往往都是针对这两个其中一个做处理，比如，你可能会挺到这个方法(dropout),dropout是在testing data不好的时候才会去使用，testing data好的时候不需要

## 训练集上效果差
### 更换激活函数（New activation function）
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-4.png)
现在你的training data performance不好的时候，是不是你在做neural的架构时设计的不好，举例来说，你可能用的activation function不够好

#  hard to get the power of deep
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-5.png)

在之前可能常用的activation function是sigmoid function,今天我们如果用sigmoid function，那么deeper usually does not imply better,这个不是overfitting

## 原因:
### vanishing Gradient problem
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-6.png)

比较靠近input 的参数最后Loss function的微分会较小，靠近output的参数对loss function的partial会很大，当你设定相同的learning rate时，靠近input layer 的参数updata会很慢，靠近output layer的参数updata会很快



![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-7.png)
咋样来想一个参数的gradinet的参数是多少呢，一个参数w对 total loss做偏微分，实际上就是对参数做一个小小的变化，对loss的影响，就可以说，这个参数gradient 的值有多大

给第一个layer的某个参数加上$\triangle w$时，对output与target之间的loss有什么样的变化。现在我们的$\triangle w$很大，通过sigmoid function时这个output会很小(一个large input，通过sigmoid function，得到small output)，每通过一次sogmoid function就会衰减一次，hidden layer很多的情况下，最后对loss 的影响非常小(对input 修改一个参数其实对output 是影响是非常小)


# 咋样去解决呢
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-8.png)
修改activation function，ReLU
input 大于0时，input 等于 output，input小于0时，output等于0

**选择这样的activation function有以下的好处**
- 比sigmoid function比较起来是比较快的
- 生物上的原因
- 无穷多的sigmoid function叠加在一起的结果(不同的bias)
- 可以处理 vanishing gradient problem

# 咋样去处理vanishing gradient problem
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-9.png)

ReLU activation function 作用于两个不同的range，一个range是当activation input大于0时，input等于output，另外一个是当activation function小于0是,output等于0

那么对那些output等于0的neural来说，对我们的network一点的影响都没。加入有个output等于0的话，你就可以把它从整个network拿掉。(下图所示)
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-10.png)
剩下的input等于output是linear时，你整个network就是a thinner linear network

我们之前说，GD递减，是通过sigmoid function，sigmoid function会把较大的input变为小的output，如果是linear的话，input等于output,你就不会出现递减的问题

我们需要的不是linear network，之所以我们用deep learning ，就是不希望我们的function不是liear我，我们需要它不是linear function，而是一个很复杂的function

如果你只对input做小小的改变，不改变neural的activation range,它是一个linear function，但是你要对input做比较大的改变，改变neural的activation range，它就不是linear function


## ReLU-variant
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-11.png)
ReLU在input小于0时，output为0，这时微分为0，你就没有办法updata你的参数，所有我们就希望在input小于0时，output有一点的值(input小于0时，output等于0.01乘以input)，这被叫做leaky ReLU

Parametric ReLU在input小于0时，output等于$\alpha z$
$\alpha$为neural的一个参数，可以通过training data学习出来，甚至每个neural都可以有不同的$\alpha$值


那么除了ReLU就没有别的activation function了吗，所以我们用Maxout来根据training data自动生成activation function

## Maxout
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-12.png)

让network自动去学它的activation function，因为activation function是自动学出来的，所有ReLU就是一种特殊的Maxout case

input是$x_1,x_2$乘以weight得到5,7,-1,1。这些值呢本来是通过ReLU或者sigmoid function等得到其他的一些value。现在在Maxout里面，在这些value group起来(哪些value被group起来是事先决定的，如上图所示)，在组里选出一个最大的值当做output(选出7和1，这是一个vector 而不是一个value)，7和1再乘以不同的weight得到不同的value，然后group，再选出max value


![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-13.png)
Maxout有办法做到跟ReLU一样的事情。

### ReLu
input乘以w,b，再经过ReLU得a

### Maxout
input中x和1乘以w和b得到$z_1$x和1乘以w和b得到$z_2$(现在假设第二组的w和b等于0，那么$z_2$等于0)，在两个中选出max得到a(如上图所示)

现在只要第一组的w和b等于第二组的w和b，那么Maxout做的事就是和ReLU是一样的

当然在Maxout选择不同的w和b做的事也是不一样的(如上图所示)，每一个Neural根据它不同的wight和bias，就可以有不同的activation function

### 面对另外一个问题，咋样去training

# Maxout-Training

![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-14.png)
max operation用方框圈起来，max operation其实在这边就是一个linear operation，只不过是在选取前一个group的element

### 把group中不是max value拿掉

![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-15.png)
没有被training到的element，那么它连接的w就不会被training到了，在做BP时，只会training在图上颜色深的实线，不会training不是max value的weight。这表面上看是一个问题，但实际上不是一个问题

当你给到不同的input时，得到的z的值是不同的，max value是不一样的，因为我们有很多笔training data，而neural structure不断的变化，实际上每一个weight都会被training


# Adaptive Learning Rate

# adagrad
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-16.png)
每一个parameter 都要有不同的learning rate，这个 Adagrd learning rate 就是用固定的learnin rate除以这个参数过去所有GD值的平方和开根号，得到新的parameter

### 我们在做deep learnning时，这个loss function可以是任何形状

![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-17.png)

你的error surface是这个形状的时候，learning rate是要能够快速的变动

在deep learning 的问题上，Adagrad可能是不够的，这时就需要RMSProp


# RMSProp
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-18.png)
一个固定的learning rate除以一个$\sigma $(在第一个时间点，$\sigma$就是第一个算出来GD的值)，在第二个时间点，你算出来一个$g^1$，$\sigma^1$(你可以去手动调一个$\alpha$值，把$\alpha$值调整的小一点，说明你倾向于相信新的gradient 告诉你的这个error surface的平滑或者陡峭的程度,)



![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-19.png)

除了learning rate的问题以外，我们在做deep learning的时候，有可能会卡在local minimize，也有可能会卡在 saddle point，甚至会卡在plateau的地方


其实在error surface上没有太多的local minimize，所以不用太担心。因为，你要是一个local minimize，你在一个dimension必须要是一个山谷的谷底，假设山谷的谷底出现的几率是P，因为我们的neural有非常多的参数(假设有1000个参数，每一个参数的dimension出现山谷的谷底就是各个P相乘)，你的Neural越大，参数越大，出现的几率越低。所以local minimize在一个很大的neural其实没有你想象的那么多

![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-20.png)


### 有一个方法可以处理下上述所说的问题

在真实的世界中，在如图所示的山坡中，把一个小球从左上角丢下，滚到plateau的地方，不会去停下来(因为有惯性)，就到了山坡处，只要不是很陡，会因为惯性的作用去翻过这个山坡，就会走到比local minimize还要好的地方，所以我们要做的事情就是要把这个惯性加到GD里面(Mometum)


### 现在复习下一般的GD
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-21.png)
选择一个初始的值，计算它的gradient，G负梯度方向乘以learning rate，得到$\theta_1$，然后继续前面的操作，一直到gradinet等于0时或者趋近于0时

## 当我们加上Momentu时
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-22.png)
我们每次移动的方向，不再只有考虑gradient，而是现在的gradient加上前一个时间点移动的方向


### 步骤
选择一个初始值$\theta……0$然后用$v^0$去记录在前一个时间点移动的方向(因为是初始值，所以第一次的前一个时间点是0)接下来去计算在$\theta^0$上的gradient，移动的方向为$v^1$。在第二个时间点，计算gradient$\theta^1$，gradient告诉我们要走红色虚线的方向(梯度的反方向)，由于惯性是绿色的方向(这个$\lambda $和learning rare一样是要调节的参数，$\lambda$会告诉你惯性的影响是多大)，现在走了一个合成的方向。以此类推...


### 运作

![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-23.png)
加上Momentum之后，每一次移动的方向是 negative gardient加上Momentum的方向(现在这个Momentum就是上一个时间点的Moveing)


现在假设我们的参数是在这个位置(左上角)，gradient建议我们往右走，现在移动到第二个黑色小球的位置，gradient建议往红色箭头的方向走，而Monentum也是会建议我们往右走(绿的箭头)，所以真正的Movement是蓝色的箭头(两个方向合起来)。现在走到local minimize的地方，gradient等于0(gradient告诉你就停在这个地方)，而Momentum告诉你是往右边的方向走，所以你的updata的参数会继续向右。如果local minimize不深的话，可以借Momentum跳出这个local minimize

## Adam
RMSPriop+Momentum

![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-24.png)



### 如果你在training data已经得到了很好的结果了，但是你在testing data上得不到很好的结果，那么接下来会有三个方法帮助解决
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-25.png)

## Early Stopping
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-26.png)

随着你的training，你的total loss会越来越小(learning rate没有设置好，total loss 变大也是有可能的)，training data和testing data的distribute是不一样的，在training data上loss逐渐减小，而在testing data上loss逐渐增大。理想上，假如你知道testing set 上的loss变化，你应该停在不是training set最小的地方，而是testing set最小的地方(如图所示)，可能training到这个地方就停下来。但是你不知道你的testing set(有label的testing set)上的error是什么。所以我们会用validation会 解决

会validation set模拟 testing set，什么时候validation set最小，你的training 会停下来

## Regularization
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-27.png)

重新去定义要去minimize的那个loss function

在原来的loss function(minimize square error, cross entropy)的基础上加一个regularization term(L2-Norm)，在做regularization时是不会加bias这一项的，加regularization的目的是为了让线更加的平滑(bias跟平滑这件事情是没有任何关系的)


![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-28.png)

在update参数的时候，其实是在update之前就已近把参数乘以一个小于1的值($\eta \lambda$都是很小的值)，这样每次都会让weight小一点————Weight Decay

Regularization这件事就是不希望参数离0太远


### regularization term当然不只是平方，也可以做L1-Norm

![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-29.png)

w是正的微分出来就是+1，w是负的微分出来就是-1，可以写为sgn(w)

每一次更新时参数时，我们一定要去减一个$\eta \lambda sgn(w^t)$值(w是正的，就是减去一个值；若w是负的，就是加上一个值，让参数变大)

L2、L1都可以让参数变小，但是有所不同的，若w是一个很大的值，L2下降的很快，很快就会变得很小，在接近0时，下降的很慢，会保留一些接近01的值；L1的话，减去一个固定的值(比较小的值)，所以下降的很慢。所以，通过L1-Norm training 出来的model，参数会有很大的值

## Dropout
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-30.png)

在traning的时候，每一次update参数之前，对network里面的每个neural(包括input)，做，这个Neural要不然被丢掉，每个neural会有p%会被丢掉，跟着的weight'也会被丢掉

![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-31.png)

做出这个sample，network structure就等于变瘦了(thinner)，然后，你在去training这个细长的network(每一次updata之前都要去做一次) 。所以每次update参数时，你拿来training network structure是不一样的。


你在training 时，performance会变的有一点差(某些neural不见了)，加上dropout，你会看到在testing set会变得有点差，但是dropout真正做的事就是让你testing 越做越好


![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-32.png)
在testing上注意两件事情，第一件事情就是在testing上不做dropout。另外一个是在dropout时，假设dropout rate在training 是p%， all weight都要乘以(1-p%)(假设dropout rate是p%，若在training上算出w=1,那么在testing 时，把w设为0.5)

### 为什么Dropout会有用
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-33.png)
training的时候，会丢掉一些neural，假如你在练习轻功的时候，你在脚上绑了一些重物(training)，实际上在战斗把重物拿下来(testing)，那么你就会变得很强

![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-34.png)


每个neural就是一个学生，在一个团队中，总是会有被dropout。你的partner会banlan的，你就想着我要好好做


为什么在testing时，dropout要乘以0.5(1-p%)
![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-35.png)
假设dropout rate是50%，training的时候，你总是会期望丢掉一半的neural。假设选定一组weight($w_1,w_2,w_3,2_4$)。testing时是没有dropout的，所以对同一组的weihgt来说，$z^' =2z$。现在怎么办，把所有的weight都乘以0.5，现在变为$z^'=z$


### dropout is a kind of ensemble

![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-36.png)

**ensemble方法**
我们有一个很大的training set，每次从training set里只sample一部分的data,然后training 很多的model(每个model可能structure不一样)，每个model可能variance很大，但是他们都是很复杂的model的话，平均起来，bias就会很小

![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-37.png)

真正在testing的时候，你已经training的很多的model了，丢一笔training data进来，通过model，得到一些结果在把这些结果平均起来，当做你最后的结果

![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-38.png)

当你做dropout是其实就是training了很多的network structure

![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-39.png)
在testing的时候，按照ensemble方法，把之前的network拿出来，然后把你的100笔data丢到network里面去，每一个network都会给你一个结果，这些结果的平均值就是最终的结果。但是实际上这些network太多了，没办法去给索引network丢一个input


所以，dropout最神奇的地方是，它告诉你，当一个完整的network不做dropout，而是把它的weight乘以(1-p%)，把你的training data丢进去，得到的output就是average的值

![image](http://ppryt2uuf.bkt.clouddn.com/chapter9-40.png)

在这个最简单的case里面，ensemble这件事情跟我们把weight乘以1/2得到一样的结果

只有是linear network才会有这样的结果


# 参考
[有道云笔记原文](http://note.youdao.com/noteshare?id=3a116a671b5300ad0976075f87c47449&sub=2658DA6C344A4ECCA0343821CCF4AD29)
