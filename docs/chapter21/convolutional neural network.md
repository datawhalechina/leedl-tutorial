![image](9ED5CE2E90AB4A2B881570821F4756FE)
我们都知道CNN常常被用在影像处理上，如果你今天来做影像处理，用一般的neural network来做影像处理，不一定要用CNN。比如说你想要做影像的分类，那么你就是training一个neural network,input一张图片，那么你就把这张图片表示成里面的pixel，也就是很长很长的vector。output就是(假如你有1000个类别，output就是1000个dimension)dimension

但是，我们现在会遇到的问题是这样的，我们在training neural network时，期待network的structure里面，每一个neural就是代表了一个最基本的classifiers
![image](7F6D36B7AE6149529FD3E88CAE615C4E)
第一层的neural就是最简单的classifiers，做的就是有没有绿色、黄色出现，有没有斜的条纹，那第二个layers做的事就是比它更复杂的东西，根据第一个layers的output，它如果看到横线，直线就是窗框的一部分，看到棕色的直条纹就是木纹，看到斜条纹加灰色可能就是轮胎的一部分。在根据第二个hidden layers的output，第三个hidden layer会做更复杂的事情。

现在我们一般直接用fully connect feedforward network来做影像处理的时候，往往我们会需要太多的参数，举例来说，这是一张100 *100的彩色图，把这个拉成一个vector，需要100 *100 3的pixel(30000 dimension)，那input vector假如是30000dimension，那这个hidden layer假设是1000个neural，那么这个参数就是有30000 *1000，那这样就太多了。那么CNN做的事就是简化neural network的架构。我们根据我们对影像的了解，某些weight用不上的，从一开始就把它滤掉。不用fully connect feedforward network，而是用比较少的参数来做影像处理这件事。所以CNN比一般的DNN是要简单的。

**先来讲一下，为什么我们有可能把一些参数拿掉(为什么可以用比较少的参数可以来做影像处理)**
#### why CNN for image
![image](1C68AAC5389B41C0AD2B0F67496FC003)

这里有几个观察，第一个是在影像处理里面，我们说第一层的hidden layer那么neural要做的事就是侦测某一种pattern。大部分的pattern其实要比整张的image还要小，对一个neural来说，假设它要知道一个image里面有没有一个pattern出现，它其实是不需要看整张image，它只要看image的一小部分，就可以决定这件事情

举例来说，我们现在有一张图片，第一个hidden layer的某一种neural的工作就是要侦测有没有鸟嘴的存在(有一些neural侦测有没有爪子的存在，有没有一些neural侦测有没有翅膀的存在等，合起来就可以侦测图片)，并不需要看整张图，我们只需要给neural看着一小红色方框的区域(鸟嘴)，就可以知道说，这个是不是一个鸟嘴(不需要去看整张图)，所以，每一个neural连接到每一个小块的区域就好了，不需要连接到整张完整的图。

![image](722ED8E84DEC4CD89DC86348BA89FD47)
第二个观察是这样子，同样的pattern在image里面，可能会出现在image不同的部分，但是代表的是同样的含义

比如说，第一图在左上角的鸟嘴，还有一张图在中央的鸟嘴，我们不需要去训练两个不同的detector，一个专门去侦测左上角的鸟嘴，一个去专门侦测中央有没有鸟嘴，这样就太冗了，所以我们要constant。我们不需要两个neural去做两组参数，我们就要求这两个neural用同一组参数，就样就可以减少参数的量

![image](4B4E7B156968435FB6DFB955EBD74F57)
第三个是，对于一个image你可以对其做subsampling,你把一个image的奇数行，偶数列的pixel拿掉，变成原来十分之一的大小，不会影响人对这张image的理解。我们就可以这样做把image变小，这样就可以减少你的参数。

#### CNN架构
![image](556094A8F6604F6DB9B78A4C0091B931)
首先input一张image，这张image会convolution，再做max pooling这件事，然后在做convolution，再做max pooling。反复无数次，反复的次数你觉得够多之后，(但是反复多少次你是要事先决定的，它就是network的架构(就像你的neural有几层一样))，然后再做flatten，再把flatten的output丢到一般fully connected feedforward network，然后得到影像辨识的结果。
![image](62560083A67146B5A5C71DDCCECF4116)

我们基于三个对影像处理的观察，所以设计了CNN这样的架构

那第一个观察是，要生成一个pattern，不要看整张的image，你只需要看image的一小部分。第二是，通用的pattern会出现在一张图片的不同的区域。第三个是，我们可以做subsampling

前面的两个property可以用convolution处理掉，最后的property可以用max pooling处理掉。

#### Convolution
![image](57C4336366684542A41D9D1CCAA7146C)

假设现在我们的network的input是一张6*6的Image，如果是黑白的，那我们就可以用一个value去描述它，1就代表有涂抹水0就代表没有涂到墨水。那在convolution layer里面，它由一堆的filter，没一组的filter就是一个maxtrix(3 *3)，这每个filter里面的参数就是network的parameter(这些parameter是要学习出来的)

每个filter如果是3* 3的detects意味着它就是再侦测一个3 *3的pattern(看3 *3的一个范围)


![image](CF97656A1506460783D9DE31981C6D7B)
这个filter是一个3* 3的matrix，把这个filter放在image的左上角，把filter的9个值和image的9个值做内积，等于3。然后移动filter的位置(移动多少是事先决定的)，移动的距离叫做stride(stride等于多少，自己来决定)，内积等于-1

![image](D06D5746F0D14396A0A9F60E8F3F9FA0)
stride=1，每次移动一格再做内(直到右下角)，结果如下图
，本来是6*6的matrix，经过convolution得到4 *4的matrix。如果你看filter的值，它斜对角的地方是1 1 1，所以它的工作就是看有有没有连出现在这张image里面。这个filter就会告诉你左上和左下出现最大的值，就代表说这个filter要侦测的pattern，出现在这张image的左上角和左下角，这件事情就考虑了propetry2(用同一个filter就可以做出来)

![image](5823C94A5BE9473CA3419B1E3DD9C4E6)
在convolution里面会有很多的filter(刚才只是一个filter的结果)，那另外的filter会有不同的参数，它也做跟filter1一摸一样的事情，在filter放到左上角再内积得到结果-1，依次类推。

你把filter2跟image做完convolution之后，你就得到了另一个4*4的matrix，红色4 *4的matrix跟蓝色的matrix合起来就叫做feature map，看你有几个filter，你就得到多少个image(你有100个filter，你就得到100个4 *4的image)

#### 刚才举例的是一张黑白的imag，若现在换成彩色
![image](A8C9678BACEE4E45A460FB65EC85D519)

彩色的image是由RGB组成的，所以，一个彩色的image就是好几个matrix叠在一起，就是一个立方体

处理彩色image，这时候filter不是一个image，也是一个立方体。如果今天是RGB表示一个pixel的话，那input就是3*6 *6，filter就是3 *3 *3.这时候你做convolution的话，就是将filter的9个值和image的9个值做内积(不是把每一个channel分开来算，而是合在一起来算，一个filter就考虑了不同颜色所代表的channel)

#### convolution和fully connected之间的关系
![image](E659015A65EA4A1C9BE43CC8471F62D5)

convolution就是fully connected layer把一些weight拿掉了。经过convolution的output其实就是一个hidden layer的neural的output。如果把这两个link在一起的话，convolution就是fully connected 拿掉一些weight的结果

![image](8267D64F77CF413C82BF53F230254DA1)
我们在做convolution时，我门filter1放到左上角(先考虑filter1)内积为3，这件事情就等同于把6* 6得matrix拉直，变成如图所示。然后你有一个neural的output是3，这个neural的output就是filter乘以(1,2,3,7,8,9,13,14,15)的pixel。那么这个neural的output(3)只连接在(1,2,3,7,8,9,13,14,15)这些pixel，这些weight就是filter的值

在fully connected中，一个neural应该是连接在所有的input，但是现在只连接了9个input，这样做就是用了比较少的参数了。

![image](A7348C52DEB64EE882A8261C1ECB58E7)

把stride=1(移动一格)做内积得到另外一个值-1，假设这个-1是另外一个neural的output，这个neural连接到input的(2,3,4，8,9,10,14，15,16)。在fully connected中这两个neural本来是有自己的weight，当我们在做convolution时，首先把neural连接的wight减少，强迫这两个neural公用一个weight。(shared weight)

#### Max pooling
![image](6812B5B74A7644B893310E87DE49A129)
![image](FFA72643A20448379AC08A31290D362B)

根据filter1得到4*4的maxtrix，根据filter2得到另一个4 *4的matrix，接下来把output4个一组，把四个value合成一个value(可以选择average也可以选择max value)，这就可以让你的imag缩小。

![image](8014C6F33E59470092D90EBAF3B4A37C)
假设我们选择四个里面的max vlaue保留下来


![image](8123607E1F084084BDB9203146414A05)
做完一个convolution和一次max pooling，从一个6 * 6的image变成了一个2 *2的image.这个2 *2的pixel的深度depend你有几个filter(你有50个filter你就有50维)，得到结果就是一个new image but smaller,一个filter就代表了一个channel。

![image](652441EBF1154240A8267E305D19E547)
这件事可以repeat很多次，通过一个convolution+ max pooling就得到a new image。再次通过convolution+max pooling的得到一个更小的image。

##### 问题：
第一次有25个filter，得到25个feature map，第二个也是由25个filter，那将做完是不是要得到`$25^2$`的feature map。其实不是这样的

假设第一层filter有2个，第二层的filter在考虑这个imput时是会考虑深度的，并不是每个channel分开考虑

#### Flatten

![image](53E0E43D2F9742AC86AF48ED684D12DA)
flatten就是feature map拉直，拉直之后就可以丢到fully connected feedforward netwwork

#### KNN in Keras
![image](AD9B77967831485C93760C26CCAC7181)
本来在DNN中input是一个vector，现在是CNN的话，会考虑image的几何空间的，所以不能给它一个vector。应该input一个tensor(高维的vector)

##### convolution
convolution2D

25__代表有25个filter，3 *3___代表filter是一个3 *3的matrix
 
1___黑白为1，彩色的图为3   ，  28*28___28 *28pixel

##### max pooling

2,2____把2*2的feature map里面的pixel拿出来，选择max value

![image](EB6AADEF5DBF409B9D3152FA7AF06591)

假设我们input一个1 *28 *28的Image，通过convplution以后得到output是25 *26 26(25个filter，通过3 *3得到26 * 26)

然后做max pooling，2*2一组选择 max value得到 25 *13 * 13

然后在做一次convolution，50*3 *3(50个filter)，得到50 *11 * 11(通过3 * 3的filter，长和宽都会减2)，再做max pooling得到 50 * 5 * 5


![image](F0FE73B4149C4C2DA97B45CE21F23560)
把50*5 *5拉直变成1250的vector，丢到fully connected network，得到output


#### CNN到底学到了什么
![image](638427D6F125472484E968EBD2C2EAB8)
##### 分析一个filter所做的事情

在第二个convolution layer里面的50个filter，每一个filter的output就是一个matrix,在这里是11*11的matrix。假设把第k个filter拿出来，是一个11 *11的matrix(如图所示)，每一个elemnet表示为`$a^k_{ij}$`(第k个filter，第i个row，第j个column)

看output11*11值全部加起来，当做现在这个filter被active的程度`$![image](B1763BB6150E464F91BB2FF22D67499F)$`

我们想知道第k个filter的作用是什么，所以找一张image，这张image可以让第k个filter被active的程度最大


假设input的image称为`$x^\ast $`现在解的问题是找一个x可以让定义的 degree of the activation`$a^k$`最大，可以用gradient ascent(update这个x，可以让degree of the activation是最大的)


![image](7FDAB9B588B2434EA6D703BA69FCEEA0)

对每一个filter找一张image，这些image有共同的特征，某种纹路在图上不断的反复，在第三张图上有很多的斜条纹，这意味着第三个filter的工作就是看图上有没有斜的条纹。所以每个filter就是determine某一种happpen(在这张图上，每一个filter就是detremine不同方向的条纹)

![image](A973F6E33AE347379060218BDEF7BC47)
在这个neural network里面每一个neural它的工作是什么。定义第j个neural它的output是`$a_j$`。接下来要做的是，找一张image x，丢到这个neural network里面去，它可以让这个`$a_j$`的值maximize

随便取9个neural出来，什么样的图丢到CNN可以让这9个neural最大active，就是这9张图

这些图跟刚才所观察到图不太一样，在刚在的filter观察到的是类似纹路的东西，那是因为每个filter考虑是图上一个小小的range。但是你在做flatten以后，每个neural的工作就是去看整张图。

![image](386D40307EB54A7D893A00281B6D02D6)
那今天我们考虑是output呢？

把某一维拿出来，找一张image让那个维度output最大(现在我们找一张image，它可以让对应在数字1的output 最大，那么它就像看起来是数字1)(这里画了9个)

machine学习到的东西跟我们是非常不一样的
[相关的paper](https://www.youtube.com/watch?v=M2IebCN9H)




#### 那我们有没有办法让图更像数字呢？
![image](F0E31EDC7B6D4CD49C1DA9B81FB01545)

一张图是不是数字我们有一些基本的假设，我们对找出的x做一些control，我们告诉machine，有些x可能会使y很大但不是数字。

对x做一些control，假设这个image里面的每个pixel用`$x_{ij}$`表示。我们希望找出的image，大部分的地方是没有涂颜色的，只有非常少的部分是有图笔画的(结果如右图所示)


#### Deep Dream

![image](E50823EF2C18470398A6BF6EC6202FA7)

如果你给machine一张image，machine会在这张image里加上它所看到的东西

将这张image丢到CNN中，把它的某一个hidden layer拿出来(vector)，然后把postitive dimension值调大，把negative dimension值调小

找一张image(modify image)用GD方法，让它在hidden layer output是你设下的target。这样做就是让CNN夸大化它所看到的东西


![image](C5B4AA61AE094DD6A796005396E52ADE)

更加强化它所看到的东西

#### Deep style

![image](F1D0B645D9C546B1B2C7F2270E32CBEF)

input一张image，让machine去修改这张图，让它有另外一张图的风格

![image](E15B8A0CD3A748D2A5B807E22DAB4CB6)

![image](85B82572ABCA4C98AE642B64C0EFEF8E)


#### More Application
![image](9BEA257F24784FE092B0588561F972CA)

任意一个neural，input一个棋盘，output是棋盘上的一个位置(下一步根据棋牌，落子的位置)。使用fully connected feedforward也可以做到


但是采用CNN的话，会得到更好的performance。把棋牌表示为19*19的matrix，当做image来看，然后output下一步output落子的位置。

![image](6C57999E86894559A4EF4736BF721B96)
使用CNN时，image必须有该有的特性，在开头就开始讲了三个观察，所以设计出了CNN这样的架构。在处理image时是特别有效的。这样的架构也同样可以用在围棋上面(因为围棋有一些特性跟影像处理是非常相似的)


在image上面，有一些pattern是要比整张image还要小的多的(鸟喙是要比整张的image要小的多)，只需要看那一小的部分就知道那是不是鸟喙。在围棋上也有同样的现象，如图所示，一个白子被三个黑子围住(这就是一个pattern)你现在只需要看这一小部分，就可以知道白子落子的地方，不需要看整个棋盘(这跟image是有同样的性质)

同样是pattern会出现在不同的range但是代表的是同样的意义，如图所示(可以出现在棋牌的左上角，也可以出现在围棋的右下角，都代表了同样的意义。)


##### 前两点都是具有image的特性想不通的第三点

我们可以对image做subsampling，拿掉奇数行偶数列pixel，变为原来的1/4的大小，也不会影响你看这张图的样子

对于围棋，当然是不能丢掉奇数行偶数列的，这样就不是一个棋牌了

#### Alpha Go的附录
![image](5939AA6FC8044CB2A0013F0CDFE2A7F2)

##### 描述Neural network structure
input是19*19 * 48的image(对于alpha go来说，它的每一个位置都用48个value来描述)。将原来19 *19的image外围补上更多0变为23 * 23的image。第一个layer是5 * 5的filter，总共是有k个filter(k=192)


Alpha GO does not use Max pooling

![image](840D95C955E94E059F70BF0552B8442C)




