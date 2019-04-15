[TOC]
### keras:Buliding a Network
![image](http://ppryt2uuf.bkt.clouddn.com/chapter8-1.png)
- Neural network是长什么样的，在keras里先宣告你的model是sequential

```
model = sequential()
```

##### 一个layer
你看要你的neural长什么样子，自己就决定长什么样子，举例，这里hidden layer 有两个layer，每个layer都有500 Neural。已经宣告了一个model，然后model.add，加一个Fully connect laye(这里用Dense表示)，然后input，output

然后增加一个activation(激活函数)，将sigmoid当做activation(也可以使用其他的当做activation)

```
model.add(activation('sigmoid'))
```
##### 下一个layer
- 这个layer的input就是上一个layer的output，不用说input是500Neural，keras自己知道

##### output
- output为10dimension
- activation为softmax

#### goodness of function(evalution)
![image](http://ppryt2uuf.bkt.clouddn.com/chapter8-2.png)
- evalution function的好坏

```
campile___编译
model.compile
```
定义一个loss是什么(不同的场合，需要不同的loss function)

```
optimizer___优化器   metrics___指标
loss = ('cateqorical crossentropy')
```
##### 3.1configuration
![image](http://ppryt2uuf.bkt.clouddn.com/chapter8-3.png)
找最好的function时，以什么样的方式来找这个function
```
model.compile = (loss = 'categorical crossentropy',
                 optimizer = 'adam')
```
- optimizer后面可以跟不同的方式，这些方式都是GD，只是用的learning rate不同，有一些machine会自己决定learning rate
##### 3.2 find the optimal network parameters
![image](http://ppryt2uuf.bkt.clouddn.com/chapter8-4.png)
- 给四个input
- x_train, y_train, batch_size, nb_epoch
- train data就是一张一张的image, laber___数字


- two dimension matrix(X_train)，第一个dimension代表你有多少个example，第二个dimension代表你有多少个pixel
- two dimension matrix(y_train)，第一个dimension代表你有多少个training example，第二个dimension代表label(黑色的为数字，从0开始计数)
##### mini-batch
![image](http://ppryt2uuf.bkt.clouddn.com/chapter8-5.png)
我们在做GD和Dp时，我们并不是真的minmize total loss,我们做的是会把train data随机分成mini-batch
- randomly initialize network parameter(跟GD一样)
- 随机选择一个batch出来,对第一个选择出来的batch里面total loss, 计算偏微分，根据`${L}'$`去update parameters
- 然后选择第二个batch ，对第一个选择出来的batch里面total loss, 计算偏微分，根据`${L}''$`去update parameters
- 直到把所有的batch都统统选过一次
- 假设今天有100个batch的话，就把这个参数updata 100次
- 把所有的batch都看过一次叫做one epoch，重复以上的过程

```
model.fit(x_train, y_train, batch_size =100, nb_epoch = 20)
```
- 这里的batch_size代表一个batch有多大(就是把100个example，放到一个batch里)
- 每个batch看过20次， 以上这个操作重复20次

#### Speed
![image](http://ppryt2uuf.bkt.clouddn.com/chapter8-6.png)
- batch-szie不同时，一个epoch所需的时间是不一样的
- batch =10相比于batch=1，较稳定
very large batch size yield worse performance，而且容易卡住

##### Speed-- why minni batch is faster than stochastic GD(为什么批量梯度下降比随机梯度下降要快)
![image](http://ppryt2uuf.bkt.clouddn.com/chapter8-7.png)
- 之前提到的矩阵计算
![image](http://ppryt2uuf.bkt.clouddn.com/chapter8-8.png)
- 拼接起来，变成一个matrix
how to use neural network(testing)
![image](http://ppryt2uuf.bkt.clouddn.com/chapter8-9.png)

