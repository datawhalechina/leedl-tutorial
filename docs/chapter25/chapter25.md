## 1-of-N Encoding



![](res/chapter25-1.png)

word embedding 是 dimension reduction 的一个广为人知的应用。如果今天要你用一个 vector 来表示 word，你会怎么做？最 difficult 的做法叫做：1-of-N Encoding。

每一个 word 用一个 vector 来表示，这个 vector 的 dimension 就是这个世界上可能有的 word 的数目。假设世界上可能有 10w 个 word，那 1-of-N Encoding 的 dimension 就是 10w 维，那每一个 word 就对应到其中一维。那 apple 就是第一维是 1 其它为 0，bag 就是第一维是 1 其它是 0，以此类推。如果你用这种方式来描述一个 word，你的每一个 word 的这个 vector 都是不一样的，所以你从这个 vector 里面没有办法得到任何的咨询，比如说：cat 跟 dog 都是动物这件事，你没有办法知道。

那怎么办呢？有一个方法叫做 word class。也就是你把不同的 word，但有同样性质的 word，把他们 class 成一群一群的，然后用那个 word 所属的 class 来表示这个 word。比如说 dog 跟 cat 都是 class1，ran jumped,walk 是 class2，flower,tree,apple 是 class3 等等。但是用 class 是不够的(少了一些 information)，比如说 class1 是动物，class2 是植物，都是属于生物，但是在 class 里面没有办法来呈现这件事情。或者说：class1 是动物，class2 是动物可以做的行为，可以看出是有一些关联的，但是用 class 没有办法来呈现出来。

所以我们需要 word Embedding：把每一个 word 都 project 到 high dimension space 上面(但是远比 1-of-N Encoding 的 dimension 要低，比如说有 10w 个 word，那 1-of-N Encoding 就是 10w 维。但是 project 到 high dimension，通常是 100 维左右)我们希望可以从这个 word Embedding 图上可以看到的结果是：类似同一个语义的词汇，在这个图上是比较接近的。而且在这个 high dimension space 里面，每一个 dimension 可能都有它特别的含义。假设我们做完 word Embedding 以后，每一个 word Embedding 的 feature vector 长这个样子，那你可能就知道说这个 dimension 代表了生物和其它东西之间的差别(横轴)，那个 dimension 就代表了跟动作有关的东西(纵轴)

## 词嵌入

![image](res/chapter25-2.png)



那怎么做 word Embedding 呢？word Embedding 是 Unsupervised。我们怎么让 machine 知道每一个词汇的含义是什么呢，你只要透过 machine 阅读大量的文章，它就可以知道每一个词汇它的 embeding feature vector 应该长什么样子。

![image](res/chapter25-3.png)



我们要做的是：learn 一个 neural network，找一个 function，你的 input 是一个词汇，output 就是那个词汇对应的 word Embedding。我们手上有的 train data 是一大堆的文字，所以我们只有 input，但是我们没有 output(我们不知道每一个 word Embedding 应该长什么样子)。所以我们要找的 function 只是单向(只知道输入，不知道输出)，这是一个 Unsupervised 的问题。

那这个问题要怎么解呢？我们之前讲过一个 deep learning base dimension reduction 的方法叫做 auto-encoder，这个问题是没有办法用 auto-encoder 来解。

![image](res/chapter25-4.png)



那怎么样找这个 word Embedding 呢？精神：你要如何了解一个词汇的含义呢，你要看它的 context(每一个词汇的含义，可以根据它的上下文来得到)

举例来说：假设机器读了一段文字：“马英九 520 宣誓就职”，它又读了一段新文字：“蔡英文 520 宣誓就职”。对机器来说，它不知道马英九和蔡英文指的是什么，但是马英九和蔡英文后面有接 520 宣誓就职。对机器来说，它只要读了大量的文章，发现马英九和蔡英文后面有类似的 context，机器就会推论说：蔡英文和马英九代表了某种有关系的物件(不知道他们是人，但机器知道说：蔡英文和马英九这两个词汇代表是同样地位的东西)

### 基于计数的词嵌入

![image](res/chapter25-5.png)



怎样用这个精神来找出这 word Embedding 呢？有两个不同体系的做法，一个做法是：count based。count based：如果我们现在有两个词汇$w_i,w_j$，他们的 word vector(用 V($w_i$,)V($w_j$来表示)，如果$w_i,w_j$它们常常在同一个文章出现，那他们的 V($w_i$,)V($w_j$)会比较接近。这个方法一个很代表性的例子叫做 Glove vector。

这个方法的原则是这样，假设我们知道$w_i$的 word vector 是$V(w_i)$,$w_j$的 word vector 是$V(w_j)$,我们可以计算 V($w_i$,)V($w_j$)它的 inner product，假设$N_{i,j}$是$w_i$,$w_i$他们在同一个 document 的次数，那我们就希望这两件事情（内积和同时出现的次数）越接近越好。你会发现说：这个概念跟我们之前将的 matrix factorozation 的概念其实是一样的

### 基于预测的词嵌入

另外一个方式是：prediction based 方法，据我所知，好像没有人很认真的比较过 prediction based 方法跟 counting based 方法有什么样非常不同的差异或者是谁优谁略。

![image](res/chapter25-6.png)



#### 具体步骤

prediction based 想法是这样子的：我们来 learn 一个 neural network，这个 neural network 做的就是 given$w_{i-1}$(w 就代表一个 word)，prediction 下一个可能出现的 word 是什么。每一个 word 可以用 1-of-N encoding 表示成一个 feature vector，所以我们要做 prediction 这件事情的话，我们就是 learning 一个 neural network，它的 input 就是$w_{i-1}$的 1-of-N encoding feature vector，它的 output 就是下一个 word$w_i$是某一个 word 的几率。也就是说，这个 model 它的 output dimension 就是 word(假设世界上有 10w 个 word，这个 model 的 output 就是 10w 维)，每一维代表了某一个 word 是下一个 word($w_i$)的几率。

假设这是一个 multiple layer perceptron neural network，那你把 feature vector 丢进去的时候，通过一些 hidden layer，就会得到 output。接下来我们把第一个 hidden layer 的 input 拿出来，第一个 dimension 是$z_1$，第二个 dimension 是$z_2$，以此类推。那我们用这个 z 就可以代表一个 word，input 不同的 1-of-N encoding，这边的 z 就会不一样。所以我们把 z 拿来代表一个词汇，就可以得到这个 vector。



![image](res/chapter25-7.png)



prediction based 的方法是怎么体现：根据一个词汇的上下文来了解一个词汇的含义这件事情呢？假设我们的 train data 里面有一个文章是“蔡英文宣誓就职”，另一和文章是“马英九宣誓就职”，在第一个句子里，蔡英文是$w_{i-1}$,宣誓就职是$w_{i}$，在另外一篇文章里面，马英九是$w_{i-1}$，宣誓就职是$w_{i}$。

你在训练这个 prediction model 的时候，不管 input 蔡英文还是马英九，你都会希望 learn 出来的结果是：宣誓就职的几率比较大的。所以你在 input 蔡英文，马英九的时候，它对应到“宣誓就职”那个词汇几率是高的。蔡英文，马英九虽然是不同的 input，但是为了让 output 的地方得到一样的 output，那你就必须让中间的 hidden layer 做一些事情。中间的 hidden layer 必须要学到说，这两个不同的词汇，必须要通过 weight 转化以后对应到同样的空间(进入 hidden layer 之前，必须把他们对应到接近的空间中，这样子我们在 output 的时候，他们才能有同样的几率)

所以当我们 learn 一个 prediction model 的时候，考虑 word context 这件事情，就自动地考虑在这个 prediction model 里面。所以我们把 prediction model 的第一个 hidden layer 拿出来，我们就可以得到我们想要找的这种 word embedding 的特性。

#### 共享参数

![image](res/chapter25-8.png)

那你可能会说：用$w_{i-1}$去 prediction$w_i$好像觉得太弱(给你一个词汇，prediction 下一个词汇，下一个词汇的可能是千千万万的)。那怎么办呢？

你可以拓展这个问题，我希望 machine learn 的是前面两个词汇($w_{i-2},w_{i-1}$)，然后 prediction 下一个词汇($w_i$)，你可以轻易的把这个 model 拓展到 n 个词汇。如果你真要 learn 这样的 word embedding 的话，你的 input 通常是至少 10 个词汇，你这样才有可能 learn 出 reasonable 的结果(只 input 一个或者两个太少了，我这里用 input 两个 word 当做例子)

注意的地方是：如果是一般的 neural network，你就把 input$w_{i-1},w_{i-2}$的 1-of-N encoding vector 把它接在一起变成一个很长的 vector，直接丢在 neural network 里面当做 input 就可以了。但是实际上你在做的时候，$w_{i-2}$的第一个 dimension 跟第一个 hidden neural 它们中间连的 weight 和$w_{i-1}$的第一个 dimension 跟第一个 hidden neural 它们中间连的 weight，这两个 weight 必须是一样的，以此类推。

为什么这样做呢，一个显而易见的理由是：如果我们不这样做，你把同一个的 word 放在$w_{i-2}$的位置跟放在$w_{i-1}$的位置，通过这个 transform 以后，它得到 embedding 就会不一样。另外一个理由是：这样做的好处是，可以减少参数量，那你就不会随着你的 context 增长，而需要更多的参数。

![image](res/chapter25-9.png)



现在假设$w_{i-2}$的 1-of-N encoding 是$x_{i-2}$,$w_{i-1}$的 1-of-N encoding 是$x_{i-1}$，那它们的长度都是 V 的绝对值。那 hidden layer 的 input 写成一个 vector z，z 的长度写成 Z 的绝对值，z 等于$x_{i-1}$乘以$W_1$加上$x_{i-2}$乘以$W_1$。现在$W^1,W^2$都是一个$\left | Z|X|V \right |$ weight matrices


我们强制$W^1=W^2=W$，所以我们今天实际在处理这个问题的时候，你可以把$x^{i-1}$跟$x^{i-2}$加起来($z= W(x_{i-1}+x_{i-2})$)。那你今天要得到一个 word vector 的时候，你就把 1-of-N encoding 乘上这个 W，就得到了这个 word embedding。

![image](res/chapter25-10.png)



在实际上，你咋样让$W^1,W^2$一样呢。做法是这样子的：假设我现在有两个 weight$w_i,w_j$，我们希望$w_i=w_j$。首先你要给$w_i,w_j$一样的 initializati，然后 update $w_i$。然后计算$w_j$对 cost function 的偏微分，然后 update$w_j$。你可能会说，$w_i,w_j$对 C 的偏微分是不一样的，再做 update 以后，那它们的值就不一样了呀。如果你只是列这样的式子，$w_i,w_j$经过一次 update 以后就不一样了。

那我们就把$w_i$对 C 的偏微分减去$w_j$对 C 的偏微分，$w_j$对 C 的偏微分减去$w_i$对 C 的偏微分，$w_i,w_j$的 update 就一样了

#### 训练

![image](res/chapter25-11.png)



那咋样训练这个 network 呢？这个训练是 Unsupe(rvised 的，所以你只需要 collect 一大堆的文字 data(爬虫)，然后就可以 train 你的 model 了。

比如说有一个句子是：“潮水退了就只知道谁没穿裤”。那你让你的 neural network input “潮水”跟“退了”，希望 output 就是“就”。所以你就希望你的 output 跟“就”的 1-of-N encoding 是 minimize cross entropy(“就”也是 1-of-N encoding 来表示的)。然后 input“退了”跟“就”，希望它的 output 跟“知道”越接近越好。然后 input“就”跟“知道”，希望它的 output 跟“谁”越接近越好。刚才讲的只是最基本的形态。

#### Various Architectures

![image](res/chapter25-12.png)



这个 prediction baed 的 model 可以有种种的变形，我目前不确定说：在这么多的变形哪一种是比较好的(在不同的 task 上互有胜负)。有一招叫做 continuous bag of word model，我们刚才说：拿前面的词汇会 prediction 接下来的词汇。那 CBOW 的意思就是说：某一个词汇的 context 去 prediction 中间的词汇($w_{i-1},w_{i+1}$去 prediction$w_i$)skin-gram 是说:我们拿中间的词汇去 prediction 接下来的 context(用$w_i$去 prediction$w_{i-1},w_{i+1}$)

假设你有读过 word vector 的文献的话，这个 neuralnetwork 不是 deep 的，它其实就是一个 hidden layer(linear hidden layer)

![image](res/chapter25-13.png)



我们知道 word vector 有一些有趣的特性，我们可以看到说你把同样类型的东西 word vector 摆在一起(Italy 跟 Rome 摆在一起，Japen 跟 Tokyo 摆在一起，他们之间是有某种固定的关系的)或者说你把动词的三态摆在一起，同一个动词的三态中间有某种固定的关系

![image](res/chapter25-14.png)



所以从这个 word vector 里面，你可以 discover word 和 word 之间的关系。还有人发现说：把两个 word vector 两两相减，然后 project 到 two dimension space 上面。如果今天是落在这个位置的话，那这连个 word vector 具有包含的关系(海豚跟白海豚相减，工人跟木匠相减，职员和售货员相减，都落在这个地方。如果一个东西属于另一个东西的话，两个相减，他们的结果是会很类似的)

![image](res/chapter25-15.png)

我们来做一些推论，我们知道 hotter 的 word vector 减掉 hot 的 word vector 会很接近 bigger 的 word vector 减掉 big 的 word vector。Rome 的 word vector 减掉 italy 的 word vector 会很接近 Berlin 的 word vector 减掉 Germany 的 word vector。King 的 word vector 减掉 queen 的 word vector 会很接近 uncle 的 word vector 减掉 aunt 的 word vector

如果有人问你说：罗马来自于意大利，那柏林来自于什么呢。机器可以回答这样的问题，怎么做呢？(我们知道 Germany vector 会很接近于 Berlin vector 减去 Rome vector 加上 Italy vector vector)假设我们不知道答案是 Germany 的话，那你要做的事情就是：计算 Berlin vector 减去 Rome vector 加上 Italy vector，然后看看它跟哪一个 vector 最接近，你可能得到的答案是 Germany

### 多语言嵌入

![image](res/chapter25-16.png)



还可以做很多事情，可以把不同语言的 word vector 拉在一起(假设你有中文的咖啡，有一个英文的 coffee，各自去 train 一组 vector，你会发现说，中文和英文 word vector 是没有任何关系的。因为在做 word vector 的时候是凭借上下文的关系，如果今天中文和英文混杂在一起，那 machine 就没有办法来判断中文和英文之间的词汇关系)。但是假如你事先知道某几个中文词汇，某几个英文词汇是对应在一起的，你先得到一组中文的 word vector，再得到一组英文的 word vector。接下来你再 learn 一个 model，它把中文和英文对应的词汇(加大对应 enlarge，下跌对应到 fall)通过这个 projection 以后，把他们 project 到 space 同一个点。

图上上面是绿色，然后下面是绿色英文，代表是已经知道对应关系中文和英文的词汇。然后你做这个 transform 以后，接下来有新的中文词汇跟新的英文词汇，你都通过 projection 把他们 project 到同一个 space 上面。你就可以知道中文的降低跟英文的 reduce 都应该落在差不多的位置，你就可以知道翻译这样的事情

### 多域嵌入



![image](res/chapter25-17.png)



在这个 word embedding 不局限于文字，你可以对影像做 embedding。举例：我们现在已经找好一组 word vector，dog vector，horse vector，auto vector，cat vector 在空间中是这个样子。接下来，你 learn 一个 model，input 一张 image，output 是跟 word vector 一样 dimension 的 vector。你会希望说，狗的 vector 分布在狗的周围，马的 vector 散布的马的周围，车辆的 vector 散布在 auto 的周围。


假设有一张新的 image 进来(它是猫，但是你不知道它是猫)，你通过同样的 projection 把它 project 这个 space 以后。神奇的是，你发现它就可能在猫的附近，那 machine 就会知道这是个猫。我们一般做影像分类的时候，你的 machine 很难去处理新增加的，它没有看过的。

如果你用这个方法的话，就算有一张 image，在 training 的时候你没有看到过的 class。比如说猫这个 image，从来都没有看过，但是猫这个 image project 到 cat 附近的话，你就会说，这张 image 叫做 cat。如果你可以做到这件事的话，就好像是 machine 阅读了大量的文章以后，它知道说：每一个词汇它是什么意思。先通过阅读大量的文章，先了解词汇之间的关系，接下来再看 image 的时候，会根据它阅读的知识去 match 每一个 image 所该对应的位置。这样就算它没有看过的东西，它也有可能把它的名字叫出来。

### 文档嵌入

![image](res/chapter25-18.png)

刚才讲的是 word embedding，也可以做 document embedding。也就不是把 word 变成一个 vector，也可以把 document 变成一个 vector

#### 语义嵌入

![image](res/chapter25-19.png)



那咋样把一个 document 变成一个 vector 呢？最简单的方法我们之前已经讲过了，就是把一个 document 变成一个 bag-of-word，然后用 auto encoding 就可以 learn 出 document semantic embedding。但光这样做是不够

### Beyond Bag of Word

![image](res/chapter25-20.png)

我们光用 bag-of-word 描述一个 document 是不够的，为什么呢？因为我们知道词汇的顺序代表了很重要的含义。举例来说，如图有两个句子“white blood cells destorying an infection”，“an infection destorying white blood cells”。如果这两句话，你看它们的 bag-of-word 是一模一样的(词汇相同)。上面这句话：白细胞消灭可传染病是 positive，下面这句话是 negative。虽然他们有相同的 bag-of-word，但是它们的语义是不同的，所以如果只是用 bag-of-word 来描述一张 document 是非常不够的，会失去很多重要的 information。

![image](res\chapter25-21.png)



列了一大堆的 reference 给大家参考，前面的三种是 unsupervised，也就是你 collect 一大堆的 document，你就可以让它自己去学。后面的几个方法是 supervised 的，在这些 document 你需要进行额外的 label。