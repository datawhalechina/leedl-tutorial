### Brief introduction of Deep learning

#### up and downs of Deep learning
![image](153A26B3D5E9466DB4EC4A871D45199B)

#### Three steps for Deep learning![image](621586212F404B9691C685CA0FF2967B)
- define a set of function(Neural Network)
- goodness of function 
- pick the best function

#### Neural Network
![image](79BC1D5D04E143B19B073C693FFAD8DA)
- 每一个logistic regression就是一个Neural Network，
- 可以用不同的方法连接Neural Network,得到不同的structures
-  Network parameter(`$\theta$`):all the weights and bias 

###### 这些neural应该咋样去接起来呢

#### Fully connect Feedforward Network 
![image](3038301C90674B29B784734A8EDC8E8C)

- Fully connect Feedforward Network是一种最常见的连接方式
- 把Neural排成两个一排一排的，每个Neural都有一个weight and bias(根据training data找出来)
- 如果weight是1和-2，bias是1，给定输入(1,-1)得出0.98，如果weight是1和-1，bias是0，给定输入(1,-1得出0.12)(依次类推)
- 可以看出一个Neural Network就是一个function


如果我们不加参数，只设定network结构，我们就定义了一个function set,设定不同的参数，就是不同的function


![image](1FE0798811344A5299B5BC3417DBF07A)
- 设定不同的parmeter，就是不同function

![image](310EDFF38D71488AB1559C45C8D32838)
- 每一个球就是Neural,在layer中，Neural是两两互相连接的
- 每一个Neural的input就是一组dimensioin，output就是整个Neural的output
- 这样就组成了一个神经网络的Deep，有很多的layers

![image](24076728DCEC48E2BCDB53C49D3E9DB8)
- Deep = many hidden layers
![image](07930CD048F44A9281CBAAA9E67B5A86)
- Residual Net需要special structure
- 简单来说，越高层，普遍规律是层数越高准确率越高


#### Matrix operation
![image](CBC1FFC8567541AC9FD80B7BA328E626)
- Neural的运作通常用Matrix operation来表示
- 把input的[1,-1]看成一个列向量乘以matrix加上bias的vector，再通过一个sigmoid function得到最后的结果
#### Neural Network
![image](5224C7B808E34556B4A22E770BC329CA)
- 把layer1的weight和bias全部集合起来为`$w^1,b^1$`，layer2的weight和bias全部集合起来为`$w^2,b^2$`，依次类推layelL
- 把input集合起来得到一个X
##### outputy等于
- `$\sigma (w^1x+b^1)=a^1$`，`$\sigma (w^2a^1+b^1)=a^2$`，依次类推
- Neural Network就是一连串的matrix的operation(matrix*vector*vector)
- 写成矩阵运算的好处就是，可以利用GPU加速

#### Output Layer
![image](05CCFE3A928042BF801D0D094C31019C)
- input layer(feature)
- hidden layer(feature extractor)
- output layer(Multi-class classifier)，将前一个layer当做feature(这个layer，是做很复杂的转化之后，变成最好的feature)

#### Example Application

![image](13B3E6903A14485EA3B2933677EC2D0F)
- 对于机器学说一个image，就是一个vector，这个vector有256的pixel，每一个pixel对应着一个dimension
- ou tput可以看成对应到每一个数学的几率

##### Handwriting Digit Recognition
![image](ACE3A66AEE9A4B8695D55010B4026F6D)
- input 256-dimension vector ,output 10 dimension vector，这个function为Neural Network
- 这个Neural Network就可以拿来当做手写数字的function
- 用GD挑一组最适合拿来做手写数字的function
- 做个design，要有多少个hidden layer，每个hidden layer要有多少个Neural

##### 如何抽feature转化到咋样design structure

### goodness of function(定义一个function的好坏)
![image](020669F624D94BBE9F1740422F7E0913)
- 计算y和`$\hat{y}$` 的 cross entropy(交叉熵)，调整参数，使cross entropy越小越好
##### 肯定是不会只有一笔example，多个example
![image](BD2A483A11D349F3B5FE9B12FD18687A)
- 所有的corss entrppy加起来，得到total loss
- 在function set 中找一个fuunction，会minmizes这个total loss(找一个组nerwork parameters`$\theta^{*}$`，然后minmizes)

##### 使用GD去minmizes这个total loss(gradient descent)![image](AA61613A23194D86AD9A39FA4D3C9D5E)
- 一组weight，先random 一下初始值，计算GD(每一个参数对total loss的偏微分)，把这些偏微分集合起来




