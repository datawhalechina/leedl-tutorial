


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-1.png)
这边举的例子是slot filling，我们希望订票系统听到用户说：“ i would like to arrive Taipei on November 2nd”，你的系统有一些slot(有一个slot叫做Destination，一个slot叫做time ofarrival)，你的系统要自动知道说这边的每一个词汇是属于哪一个slot，那这个问题要怎样解呢。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-2.png)
这个问题你当然可以使用一个feedforward network来解，也就是说我叠一个feedforward network，input是一个词汇(把Taopei变成一个vector)丢到这个neural network里面去(你要把一个词汇丢到一个neural network里面去，就必须把它变成一个向量来表示)。那咋样把一个词汇用向量来表示呢？

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-3.png)
1-of-encoding将词汇可以变为vector。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-4.png)
Beyond 1-of-N encoding方法，比如说你只是用1-of-N-encoding来描述一个词汇的话你会遇到一些问题，因为有很多词汇你可能都没有见过，所以你需要在1-of-N encoding里面多加dimension，这个dimension代表other。然后所有的词汇，如果它不是在我们词言有的词汇就归类到other里面去(Gandalf,Sauron归类到other里面去)。你可以用每一个词汇的字母来表示它的vector，比如说，你的词汇是apple，apple里面有出现app、ppl、ple，那在这个dimension里面对应到1,而其他都为0。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-5.png)
把词汇表示为vector，把这个vector丢到feedforward network里面去，在这个task里面，你就希望你的output是一个probability distribution。这个probability distribution代表着我们现在input这词汇属于每一个slot的几率




但是光只有这个是不够的，feedforward network是没有办法slot这个probability。为什么呢，假设现在有一个使用者说：“arrive Taipei on November 2nd”(arrive-other,Taipei-dest, on-other,November-time,2nd-time)。那现在有人说:"leave Taipei on November 2nd"，这时候Taipei就变成了“place of departure”，它应该是出发地而不是目的地。但是对于neural network来说，input一样的东西output就应该是一样的东西(input  "Taipei"，output要么是destination几率最高，要么就是time of departure几率最高)，你没有办法一会让出发地的几率最高，一会让它目的地几率最高。这个肿么办呢？这时候就希望我们的neural network是有记忆力的。如果今天我们的neural network是有记忆力的，它记得它看过红色的Taipei之前它就已经看过arrive这个词汇；它记得它看过绿色之前，它就已经看过leave这个词汇，它就可以根据上下文产生不同的output。如果让我们的neural network是有记忆力的话，它就可以解决input不同的词汇，output不同的问题。



![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-6.png)
这种有记忆的neural network就叫做Recurrent Neural network(RNN)。在RNN里面，每一次hidden layer的neural产生output的时候，这个output会被存到memory里去(用蓝色方块表示memory)。那一次当有input时，这些neural不只是考虑input$x_1,x_2$，还会考虑存到memory里的值。对它来说除了$x_1,x_2$以外，这些存在memory里的值$a_1,a_2$也会影响它的output。


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-7.png)
举个例子，假设我们现在图上这个neural，它所有的weight都是1，neural没有任何的bias。假设所有的activation function都是linear(这样可以不要让计算太复杂)。现在假设我们的input 是sequence$\begin{bmatrix}
1\\ 
1
\end{bmatrix}\begin{bmatrix}
1\\ 
1
\end{bmatrix}\begin{bmatrix}
2\\ 
2
\end{bmatrix}$

把这个sequenceinput到neural里面去会发生什么事呢？在你开始要使用这个Recurrent Neural Network的时候，你必须要给memory初始值(假设他还没有放进任何东西之前，memory里面的值是0)

现在输入第一个$\begin{bmatrix}
1\\ 
1
\end{bmatrix}$，接下来对发生什么事呢？，对左边的那个neural来说(第一个hidden layer)，它除了接到input的$\begin{bmatrix}
1\\ 
1
\end{bmatrix}$还接到了memory(0跟0)，output就是2(所有的weight都是1)，右边也是一样output为2。第二层hidden laeyer output为4。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-8.png)
接下来Recurrent Neural Networ会将绿色neural的output存在memory里去，所以memory里面的值被update为2。

接下来再输入$\begin{bmatrix}
1\\ 
1
\end{bmatrix}$，接下来绿色的neural输入有四个
$\begin{bmatrix}
1\\ 
1
\end{bmatrix}\begin{bmatrix}
2\\ 
2
\end{bmatrix}$，output为$\begin{bmatrix}
6\\ 
6
\end{bmatrix}(weight=1)$，第二层的neural output为$\begin{bmatrix}
12\\ 
12
\end{bmatrix}$。

所以对Recurrent Neural Networ来说，你就算input一样的东西，它的output是可能不一样了(因为有memory)

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-9.png)

现在$\begin{bmatrix}
6\\ 
6
\end{bmatrix}$存到memory里去，接下来input是$\begin{bmatrix}
2\\ 
2
\end{bmatrix}$，output为$\begin{bmatrix}
16\\ 
16
\end{bmatrix}$,第二层hidden layer为$\begin{bmatrix}
32\\ 
32
\end{bmatrix}$


那在做Recurrent Neural Networ时，有一件很重要的事情就是这个input sequence调换顺序之后output不同(Recurrent Neural Networ里，它会考虑sequence的order)


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-10.png)

今天我们要用Recurrent Neural Networ处理slot filling这件事，就像是这样，使用者说：“arrive Taipei on November 2nd”，arrive就变成了一个vector丢到neural network里面去，neural network的hidden layer的output写成$a^1$($a^1$是一排neural的output，是一个vector)，$a^1$产生$y^1$,$y^1$就是“arrive”属于每一个slot filling的几率。接下来$a^1$会被存到memory里面去，"Taipei会变为input"，这个hidden layer会同时考虑“Taipei”这个input和存在memory里面的$a^1$,得到$a^2$，根据$a^2$得到$y^2$，$y^2$是属于每一个slot filling的几率。以此类推($a^3$得到$y^2$)。

有人看到这里，说这是有三个neural，这个不是三个neural，这是同一个neural在三个不同的时间点被使用了三次。(我这边用同样的weight用同样的颜色表示)

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-11.png)
那所以我们有了memory以后，刚才我们讲了输入同一个词汇，我们希望output不同的问题就有可能被解决。比如说，同样是输入“Taipei”这个词汇，但是因为红色“Taipei”前接了“leave”，绿色“Taipei”前接了“arrive”(因为“leave”和“arrive”的vector不一样，所以hidden layer的output会不同)，所以存在memory里面的值会不同。现在虽然$x_2$的值是一样的，因为存在memory里面的值不同，所以hidden layer的output会不一样，所以最后的output也就会不一样。这是Recurrent Neural Networ的基本概念。



![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-12.png)
Recurrent Neural Networ的架构是可以任意设计的，比如说，它当然是deep(刚才我们看到的Recurrent Neural Networ它只有一个hidden layer)，当然它也可以是deep Recurrent Neural Networ。


比如说，我们把$x^t$丢进去之后，它可以通过一个hidden layer，再通过第二个hidden layer，以此类推(通过很多的hidden layer)才得到最后的output。每一个hidden layer的output都会被存在memory里面，在下一个时间点的时候，每一个hidden layer会把前一个时间点存的值再读出来，以此类推最后得到output，这个process会一直持续下去。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-13.png)
Recurrent Neural Networ会有不同的变形，我们刚才讲的是Elman network。(如果我们今天把hidden layer的值存起来，在下一个时间点在读出来)。还有另外一种叫做Jordan network，Jordan network存的是整个network output的值，它把output值在下一个时间点在读进来(把output存到memory里)。传说Jordan network会得到好的performance。

Elman network是没有target，很难控制说它能学到什么hidden layer information(学到什么放到memory里)，但是Jordan network是有target，今天我们比较很清楚我们放在memory里是什么样的东西。


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-14..png)
Recurrent Neural Networ还可以是双向，什么意思呢？我们刚才Recurrent Neural Networ你input一个句子的话，它就是从句首一直读到句尾。假设句子里的每一个词汇我们都有$x^t$表示它。他就是先读$x^t$在读$x^{t+1}$在读$x^{t+2}$。但是它的读取方向也可以是反过来的，它可以先读$x^{t+2}$，再读$x^{t+1}$，再读$x^{t}$。你可以同时train一个正向的Recurrent Neural Network，又可以train一个逆向的Recurrent Neural Network，然后把这两个Recurrent Neural Network的hidden layer拿出来，都接给一个output layer得到最后的$y^t$。所以你把正向的network在input$x^t$的时候跟逆向的network在input$x^t$时，都丢到output layer产生$y^t$，然后产生$y^{t+1}$,$y^{t+2}$,以此类推。用Bidirectional neural network的好处是，neural在产生output的时候，它看的范围是比较广的。如果你只有正向的network，再产生$y^t$，$y^{t+1}$的时候，你的neural只看过$x_1$到$x^{t+1}$的input。但是我们今天是Bidirectional neural network，在产生$y^{t+1}$的时候，你的network不只是看过
$x_1$,到$x_^{t+1}$所有的input，它也看了从句尾到$x^{t+1}$的input。那network就等于整个input的sequence。假设你今天考虑的是slot filling的话，你的network就等于看了整个sentence后，才决定每一个词汇的slot应该是什么。这样会比看sentence的一半还要得到更好的performance。


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-15.png)
那我们刚才讲的Recurrent Neural Network其实是Recurrent Neural Network最simpleness的版本。

那我们刚才讲的memory是最单纯的，我们可以随时把值存进去，也可以把值读出来。但现在最常用的memory称之为Long Short-term Memory(长时间的短期记忆)，简写LSTM.这个Long Short-term Memor是比较复杂的。

这个Long Short-term Memor是有三个gate，当外界某个neural的output想要被写到memory cell里面的时候，必须通过一个input Gate，那这个input Gate要被打开的时候，你才能把值写到memory cell里面去，如果把这个关起来的话，就没有办法把值写进去。至于input Gate是打开还是关起来，这个是neural network自己学的(它可以自己学说，它什么时候要把input Gate打开，什么时候要把input Gate关起来)。那么输出的地方也有一个output Gate，这个output Gate会决定说，外界其他的neural可不可以从这个memory里面把值读出来(把output Gate关闭的时候是没有办法把值读出来，output Gate打开的时候，才可以把值读出来)。那跟input Gate一样，output Gate什么时候打开什么时候关闭，network是自己学到的。那第三个gate叫做forget Gate，forget Gate决定说：什么时候memory cell要把过去记得的东西忘掉。这个forget Gate什么时候会把存在memory的值忘掉，什么时候会把存在memory里面的值继续保留下来)，这也是network自己学到的。

那整个LSTM你可以看成，它有四个input 1个output，这四个input 一个是想要被存在memory cell的值(但它不一定存的进去)跟操控input Gate的讯号跟操控output Gate的讯号跟操控forget Gate的讯号，有着四个input但它会得到一个output


这个“-”应该在short-term中间，是长时间的短期记忆。想想我们之前看的Recurrent Neural Network，它的memory在每一个时间点都会被洗掉，只要有新的input进来，每一个时间点都会把memory
洗掉，所以的short-term是非常short的，但如果是Long Short-term Memory，它记得会比较久一点(只要forget Gate不要决定要忘记，它的值就会被存起来)。



![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-16.png)
这个memory cell更仔细来看它的formulation，它长的像这样。

底下这个是外界传入cell的input，还有input gate,forget gate,output gate。现在我们假设要存到被叫做z，操控input gate叫做$z_i$,操控forget gate叫做$z_f$，操控output gate叫做$z_o$，综合这些东西会得到一个output 记为a。假设cell里面
有这四个输入之前，它里面已经存了值c。

假设要输入的部分为z，那三个gate分别是由$z_i$,$z_f$,$z_0$所操控的。那output a会长什么样子的呢。我们把z通过activation function得到g(z)，那$z_i$通过另外一个activation function得到$f(z_i)$($z_i$,$z_f$,$z_0$通过的activation function 通常我们会选择sigmoid function)，选择sigmoid function的意义是它的值是介在0到1之间的。这个0到1之间的值代表了这个gate被打开的程度(如果这个f的output都是activation function的output都是1，表示为被打开的状态，反之代表这个gate是关起来的)。

那接下来，把$g(z)$乘以$f(z_i)$得到$g(z)f(z_i)$，接下来把存到memory里面的值c乘以$f(z_f)$得到c$f(z_f)$，然后加起来($c^{'}=g(z)f(z_i)+cf(z_f)$)，那么$c^{'}$就是重新存到memory里面的值。所以根据目前的运算说，这个$f(z_i)$cortrol这个$g(z)$，可不可以输入一个关卡(假设输入$f(z_i)$0，那$g(z)f(z_i)$就等于0，那就好像是没有输入一样，如果$f(z_i)$等于1就等于是把$g(z)$当做输入)
。那这个$f(z_f)$决定说：我们要不要把存在memory的值洗掉假设$f(z_f)$为1(forget gate 开启的时候),这时候c会直接通过(就是说把之前的值还会记得)。如果$f(z_f)$等于0(forget gate关闭的时候)$cf(z_f)$等于0。然后把这个两个值加起来($c^{'}=g(z)f(z_i)+cf(z_f)$)写到memory里面得到$c^{'}$。这个forget gate的开关是跟我们的直觉是相反的，那这个forget gate打开的时候代表的是记得，关闭的时候代表的是遗忘。那这个$c^{'}$通过$h(c^{'})$，将$h(c^{'})$乘以$f(z_o)$得到$a = f(c^{'}f(z_o)$(output gate受$f(z_o)$所操控，$f(z_o)$等于1的话，就说明$h(c^{'})$能通过，$f(z_o)$等于0的话，说明memory里面存在的值没有办法通过output gate被读取出来)


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-17.png)
也许这样，你还不是很清楚，我打算举一个LSTM例子。我们的network里面只有一个LSTM的cell，那我们的input都是三维的vector，output都是一维的output。那这三维的vector跟output还有memory的关系是这样的。假设第二个dimension$x_2$的值是1时，$x_1$的值就会被写到memory里，假设$x_2$的值是-1时，就会reset the memory，假设$x_3$的值为1时，你才会把output打开才能看到输出。

假设我们原来存到memory里面的值是0，当第二个dimension$x_2$的值是1时，3会被存到memory里面去。第四个dimension的$x_2$等于，所以4会被存到memory里面去，所以会得到7。第六个dimension的$x_3$等于1，这时候7会被输出。第七个dimension的$x_2$的值为-1，memory里面的值会被洗掉变为0。第八个dimension的$x_2$的值为1，所以把6存进去，因为$x_3$的值为1，所以把6输出。



![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-18.png)
那我们就做一下实际的运算，这个是一个memory cell。这四个input gate是这样来的：input的三维vector乘以linear transform以后所得到的结果($x_1$,$x_2$,$x_3$乘以权重再加上bias)，这些权重和bias是哪些值是通过train data用GD学到的。 假设我已经知道这些值是多少了，那用这样的输入会得到什么样的输出。那我们就实际的运算一下。

在实际运算之前，我们先根据它的input，参数分析下可能会得到的结果。底下这个外界传入的cell，$x_1$乘以1，其他的vector乘以0，所以就直接把$x_1$当做输入。在input gate时，$x_2$乘以100，bias乘以-10(假设$x_2$是没有值的话，通常input gate是关闭的(bias等于-10)，若$x_2$的值大于1的话，结果会是一个正值，代表input gate会被打开) 。forget gate通常会被打开的，因为他的bias等于10(它平常会一直记得东西)，只有当$x_2$的值为一个很大的负值时，才会把forget gate关起来。output gate平常是被关闭的，因为bias是一个很大的负值，若$x_3$有一个很大的正值的话，压过bias把output打开。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-19.png)
接下来，我们实际的input一下看看。我们假设g和h都是linear(因为这样计算会比较方便)。假设存到memory里面的初始值是0，我们input第一个vector(3,1,0),input这边3*1=3，这边输入的是的值为3。input gate这边($1 *100-10\approx 1$)是被打开(input gate约等于1)。($g(z) *f(z_i)=3$)。forget gate($1 *100+10\approx 1$)是被打开的(forget gate约等于1)。现在0 *1+3=3($c^{'}=g(z)f(z_i)+cf(z_f)$)，所以存到memory里面的现在为3。output gate(-10)是被关起来的，所以3无关通过，所以输出值为0。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-20.png)
接下来input(4,1,0),传入input的值为4，input gate会被打开，forget gate也会被打开，所以memory里面存的值等于7(3+4=7)，output gate仍然会被关闭的，所以7没有办法被输出，所以整个memory的输出为0。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-21.png)

接下来input(2,0,0),传入input的值为2，input gate关闭(\approx 0),input被input gate给挡住了(0 *2=0),forget gate打开(10)。原来memory里面的值还是7(1 *7+0=7).output gate仍然为0，所以没有办法输出，所以整个output还是0。



![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-22.png)
接下来input(1,0,1),传入input的值为1,input gate是关闭的，forget gate是打开的，memory里面存的值不变，output gate被打开，整个output为7(memory里面存的7会被读取出来)


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-23.png)
最后input(3,-1,0),传入input的值为3，input gate 关闭，forget gate关闭，memory里面的值会被洗掉变为0，output gate关闭，所以整个output为0。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-24.png)
你可能会想这个跟我们的neural network有什么样的关系呢。你可以这样想，在我们原来的neural network里面，我们会有很多的neural，我们会把input乘以不同的weight当做不同neural的输入，每一个neural都是一个function，输入一个值然后输出一个值。但是如果是LSTM的话，其实你只要把LSTM那么memory的cell想成是一个neural就好了。所以我们今天要用一个LSTM的neural，你做的事情其实就是原来简单的neural换成LSTM 

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-25.png)

你做的事情其实就是原来简单的neural换成LSTM 。现在的input($x_1,x_2$)会乘以不同的weight当做LSTM不同的输入(假设我们这个hidden layer只有两个neural，但实际上是有很多的neural)。input($x_1,x_2$)会乘以不同的weight会去操控output gate，乘以不同的weight操控input gate，乘以不同的weight当做底下的input，乘以不同的weight当做forget gate。第二个LSTM也是一样的。所以LSTM是有四个input跟一个output，对于LSTM来说，这四个input是不一样的。在原来的neural network里是一个input一个output。在LSTM里面它需要四个input，它才能产生一个output。

LSTM因为需要四个input，而且四个input都是不一样，所以LSTM需要的参数量(假设你现在用的neural的数目跟LSTM是一样的)是一般neural network的四倍。这个跟Recurrent Neural Network 的关系是什么，这个看起来好像不一样，所以我们要画另外一张图来表示。


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-26.png)

假设我们现在有一整排的neural(LSTM)，这些LSTM里面的memory都存了一个值，把所有的值接起来就变成了vector，写为
$c^{t-1}$(一个值就代表一个dimension)。现在在时间点t，input一个vector$x^t$，这个vector首先会乘上一matrix(一个linear transform变成一个vector z,z这个vector的dimension就代表了操控每一个LSTM的input(z这个dimension正好就是LSTM memory cell的数目)。z的第一维就丢给第一个cell(以此类推)

这个$x^t$会乘上另外的一个transform得到$z^i$，然后这个$z^i$的dimension也跟cell的数目一样，$z^i$的每一个dimension都会去操控input gate(forget gate 跟output gate也都是一样，这里就不在赘述)。所以我们把$x^t$乘以四个不同的transform得到四个不同的vector，四个vector的dimension跟cell的数目一样，这四个vector合起来就会去操控这些memory cell运作。



一个memory cell就长这样，现在input分别就是$z$,$z^i$,$z^o$,$z^f$(都是vector)，丢到cell里面的值其实是vector的一个dimension，因为每一个cell input的dimension都是不一样的，所以每一个cell input的值都会是不一样。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-27.png)
所以的cell是可以共同一起被运算的，咋样共同一起被运算呢？我们说，$z^i$通过activation function跟z相乘，$z^f$通过activation function跟之前存在cell里面的值相乘，然后将$z$跟$z^i$相乘的值加上$z^f$跟$c^{t-1}$，$z^o$通过activation function的结果output，跟之前相加的结果再相乘，最后就得到了output$y^t$

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-28.png)
之前那个相加以后的结果就是memory里面存放的值$c^t$，这个process反复的进行，在下一个时间点input$x^{t+1}$，把z跟input gate相乘，把forget gate跟存在memory里面的值相乘，然后将前面两个值再相加起来，在乘上output gate的值，然后得到下一个时间点的输出$y^{t+1}$




你可能认为说这很复杂了，但是这不是LSTM的最终形态，真正的LSTM,会把上一个时间的输出接进来，当做下一个时间的input，也就说下一个时间点操控这些gate的值不是只看那个时间点的input$x^t$，还看前一个时间点的output$h^t$。其实还不止这样，还会加一个东西叫做“peephole”，这个peephole就是把存在memory cell里面的值也拉过来。那操控LSTM四个gate的时候，你是同时考虑了$x^{t+1},h^t,c^t$，你把这三个vector并在一起乘上不同的transform得到四个不同的vector再去操控LSTM。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-29.png)
LSTM通常不会只有一层，若有五六层的话。大概是这个样子。每一个第一次看这个的人，反映都会很难受。现在还是 quite standard now，当有一个人说我用RNN做了什么，你不要去问他为什么不用LSTM,因为他其实就是用了LSTM。现在当你说，你在做RNN的时候，其实你指的就用LSTM。

GRU是LSTM稍微简化的版本，它只有两个gate，虽然少了一个gate，但是performance跟LSTM差不多(少了1/3的参数，也是比较不容易overfitting)。如果你要用这堂课最开始讲的那种RNN，你要说是simple RNN才行。

Recurrent Neural 
Network这种架构如何做learning呢?


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-30.png)
如果要做learning的话，你要定义一个cost function来evaluate你的model是好还是不好，选一个parameter要让你的loss 最小。那在Recurrent Neural 
Network里面，你会咋样定义这个loss呢，下面我们先不写算式，先直接举个例子。

假设我们现在做的事情是slot filling，那你会有train data，那这个train data是说:我给你一些sentence，你要给sentence一些label，告诉machine说第一个word它是属于other slot，“Taipei是”Destination slot,"on"属于other
slot，“November”和“2nd”属于time slot，然后接下来你希望说：你的cost咋样定义呢。那“arrive”丢到Recurrent Neural Network的时候，Recurrent Neural Network会得到一个output $y^1$,接下来这个$y^1$会看它的reference vector算它的cross entropy。你会希望说，如果我们丢进去的是“arrive”，那他的reference vector应该对应到other slot的dimension(其他为0)，这个reference vector的长度就是slot的数目(这样四十个slot，reference vector的dimension就是40)，那input的这个word对应到other slot的话，那对应到other slot dimension为1,其它为0。

那现在把“Taipei”丢进去之后，因为“Taipei”属于destination slot,就希望说把“Taipei”丢进去的话，$y^2$它要跟reference vector距离越近越好。那$y^2$的reference vector是对应到destination slot是1，其它为0.

那这边注意的事情就是，你在丢$x_2$之前，你一定要丢$x_1$(在丢“Taipei”之前先把“arrive丢近期”)，不然你就不知道存到memory里面的值是多少。所以在做train的时候，你也不能够把这些word打散来看，word sentence仍然要当做一个整体来看。把“on”丢进去，reference vector对应的other是1，其它是0.

然后你的cost就是，每一个时间点的output跟reference vector的cross entropy的和就是你要minimize的对象。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-31.png)
有了这个loss function以后，train要肿么做呢，train其实也是用GD。也就是说我们现在定义出了loss function(L)，我要update这个neural里面的某个参数w，要肿么做呢，就是计算对w的偏微分，偏微分计算出来以后，就用GD的方法去update里面的参数。在讲feedforward neural的时候，我们说GD用在feedforward neural里面你要用一个有效率的算法叫做Backpropagation。那Recurrent Neural Network里面，为了要计算方便，所以也有开发一套算法是Backpropagation的进阶版，叫做BPTT。它跟Backpropagation其实是很类似的，只是Recurrent Neural Network它是在sentence sequence上运作，所以BPTT它要考虑时间上的information。


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-32.png)
而不信的是，RNN的training是比较困难的。一般而言，你在做training的时候，你会期待，你的learning curve是像蓝色这条线，这边的纵轴是total loss，横轴是epoch的数目，你会希望说：随着epoch的数目越来越多，随着参数不断的update，loss会慢慢的下降最后趋向收敛。但是不幸的是你在训练Recurrent Neural Network的时候，你有时候会看到绿色这条线。如果你是第一次trai Recurrent Neural Network，你看到绿色这条learning curve非常剧烈的抖动，然后抖到某个地方，这时候你会有什么想法，我相信你会：这程序有bug啊。

小故事


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-33.png)
分析了下RNN的性质，它发现说RNN的error surface是total loss的变化是非常陡峭的/崎岖的(error surface有一些地方非常的平坦，一些地方非常的陡峭，就像是悬崖峭壁一样)，纵轴是total loss，x和y轴代表是参数。这样会告诉什么样的问题呢？假设你从橙色的点当做你的初始点，用GD开始调整你的参数(updata你的参数，可能会跳过一个悬崖，这时候你的loss会突然爆长，loss会非常上下剧烈的震荡)。有时候你可能会遇到更惨的状况，就是以正好你一脚踩到这个悬崖上，会发生这样的事情，因为在悬崖上的gradient很大，之前的gradient会很小，所以你措手不及，因为之前gradient很小，所以你可能把learning rate调的比较大。很大的gradient乘上很大的learning rate结果参数就update很多，整个参数就飞出去了。

用工程的思想来解决，这一招蛮关键的，在很长的一段时间，只有他的code可以把RNN的model给train出来。在他的博士论文才给出来。

这一招就是clipping(当gradient大于某一个(没听太懂)，不要让它超过(没听懂))，当gradient大于15时，让gradient等于15结束。因为gradient不会太大，所以你要做clipping的时候，就算是踩着这个悬崖上，也不飞出来，会飞到一个比较近的地方，这样你还可以继续做你得RNN的training。

问题：为什么RNN会有这种奇特的特性。有人会说，是不是来自sigmoid function，我们之前讲过Relu activation function的时候，讲过一个问题gradient vanish，这个问题是从sigmoid function来的，RNN会有很平滑的error surface是因为来自于gradient vanish，这问题我是不认同的。等一下来看这个问题是来自sigmoid function，你换成Relu去解决这个问题就不是这个问题了。跟大家讲个密码，一般在train neural network时，一般很小用Relu来当做activation function。为什么呢？其实你把sigmoid function换成Relu，其实在RNN performance通常是比较差的。所以activation function并不是这里的关键点。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-34.png)
如果说我们今天讲BPTT，你可能会从式子更直观的看出为什么会有这个问题。那今天我们没有讲BPTT。没有关系，我们有更直观的方法来知道一个gradient的大小。

你把某一个参数做小小的变化，看它对network output的变化有多大，你就可以测出这个参数的gradient的大小。

举一个很简单的例子，只有一个neural，这个neural是linear。input没有bias，input的weight是1，output的weight也是1，transition的wight是w。也就是说从memory接到neural 的input的weight是w。


现在我假设给neural的输入是(1,0,0,0)，那这个neural的output会长什么样子呢？比如说，neural在最后一个时间点(1000个output值是$w^{999}$)。

现在假设w是我们要learn的参数，我们想要知道它的gradient，所以是知道当我们改变w的值时候，对neural的output有多大的影响。现在假设w=1，那现在$y^{1000}=1$，假设w=1.01，$y^{1000}\approx 20000$，这个就跟蝴蝶效应一样，w有一点小小的变化，会对它的output影响是非常大的。所以w有很大的gradient。有很大的gradient也并没有，我们把learning rate设小一点就好了。但我们把w设为0.99，那$y^{1000}\approx0$，那如果把w设为0.01，那$y^{1000}\approx0$。也就是说在1的这个地方有很大的gradient，但是在0.99这个地方就突然变得非常非常的小，这个时候你就需要一个很大的learning rate。设置learning rate很麻烦，你的error surface很崎岖，你的gardient是时大时小的，在非常小的区域内，gradient有很多的变化。从这个例子你可以看出来说，为什么RNN会有问题，RNN training的问题其实来自它把同样的东西在transition的时候反复使用。所以这个w只要一有变化，它完全由可能没有造成任何影响，一旦造成影响，影响都是天崩地裂的(所以gradient会很大，gradient会很小)。

所以RNN不好训练的原因不是来自activation function而是来自于它有sentence sequence同样的weight在不同的时间点被反复的使用。


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-35.png)
有什么样的技巧可以告诉我们可以解决这个问题呢？其实广泛被使用的技巧就是LSTM，LSTM可以让你的error surface不要那么崎岖。它可以做到的事情是，它会把那些平坦的地方拿掉，解决gradient vanish的问题，不会解决gradient explode的问题。有些地方还是非常的崎岖的(有些地方仍然是变化非常剧烈的，但是不会有特别平坦的地方)。

如果你要做LSTM时，有些地方变化的很剧烈，所以当你做LSTM的时候，你可以放心的把你的learning rate设置的小一点。

那为什么LSTM 可以can deal with gradient vanishing的问题呢，为什么可以避免gradient特别小呢。

我听说某人在面试某国际大厂的时候被问到这个问题，那这个问题肿么样答比较好呢(问题：为什么我们把RNN换成LSTM)。如果你的答案是LSTM比较潮，LSTM比较复杂，这个就太弱了。正在的理由就是LSTM可以handle gradient vanishing的问题。接下里面试官说：为什么LSTM会handle gradient vanishing的问题呢？用这边的式子回答看看，若考试在碰到这样的问题时，你就可以回答了。

RNN跟LSTM在面对memory的时候，它处理的operation其实是不一样的。你想想看，在RNN里面，在每一个时间点，memory里面的值都是会被洗掉，在每一个时间点，neural的output都要input里面去，所以在每一个时间点，memory里面的值都是会被覆盖掉。但是在LSTM里面不一样，它是把原来memory里面的值乘上一个值再把input的值加起来放到cell里面。所以它的memory input是相加的。所以今天它和RNN不同的是，如果今天你的weight可以影响到memory里面的值的话，一旦发生影响会永远都存在。不像RNN在每个时间点的值都会被format掉，所以只要这个影响被format掉它就消失了。但是在LSTM里面，一旦对memory造成影响，那影响一直会被留着(除非forget gate要把memory的值洗掉)，不然memory一旦有改变，只会把新的东西加进来，不会把原来的值洗掉，所以它不会有gradient vanishing的问题

那你想说们现在有forget gate可能会把memory的值洗掉。其实LSTM的第一个版本其实就是为了解决gradient vanishing的问题，所以它是没有forget gate，forget gate是后来才加上去的。甚至，现在有个传言是：你在训练LSTM的时候，你要给forget gate特别大的bias，你要确保forget gate在多数的情况下都是开启的，只要少数的情况是关闭的

那现在有另外一个版本用gate操控memory cell，叫做Gates Recurrent Unit(GRU)，LSTM有三个Gate，而GRU有两个gate，所以GRU需要的参数是比较少的。因为它需要的参数量比较少，所以它在training的时候是比较快的。如果你今天在train LSTM，你觉得overfitting的情况很严重，你可以试下GRU。GRU的精神就是：旧的不去，新的不来。它会把input gate跟forget gate联动起来，也就是说当input gate打开的时候，forget gate会自动的关闭(format存在memory里面的值)，当forget gate没有要format里面的值，input gate就会被关起来。也就是说你要把memory里面的值清掉，才能把新的值放进来。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-36.png)
其实还有其他的technique是来handle gradient vanishing的问题。比如说clockwise RNN或者说是Structurally Constrained Recurrent Network (SCRN)等等。

有一个蛮有趣的paper是这样的：一般的RNN用identity matrix来initialized transformation weight+ReLU activaton function它可以得到很好的performance。刚才不是说用ReLU的performance会比较呀，如果你说一般train的方法initiaed weight是(这个单词没懂)，那ReLU跟sigmoid function来比的话，sigmoid performance 会比较好。但是你今天用了identity matrix的话，这时候用ReLU performance会比较好。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-37.png)
其实RNN有很多的application，前面举得那个solt filling的例子。我们假设input跟output的数目是一样的，也就是说input有几个word，我们就给每一个word slat label。那其实RNN可以做到更复杂的事情



![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-38.png)
那其实RNN可以做到更复杂的事情，比如说input是一个sequence，output是一个vector，这有什么应用呢。比如说，你可以做sentiment analysis。sentiment analysis现在有很多的(这个单词没懂)

某家公司想要知道，他们的产品在网上的评价是positive 还是negative。他们可能会写一个爬虫，把跟他们产品有关的文章都爬下来。那这一篇一篇的看太累了，所以你可以用一个machine learning 的方法learn一个classifier去说哪些document是正向的，哪些document是负向的。或者在电影上，sentiment analysis所做的事就是给machine 看很多的文章，然后machine要自动的说，哪些文章是正类，哪些文章是负类。肿么样让machine做到这件事情呢？你就是learning一个Recurrent Neural Network，这个input是(这个单词不懂)sequence，然后Recurrent Neural Network把这个sequence读过一遍。在最后一个时间点，把hidden layer拿出来，在通过几个transform，然后你就可以得到最后的sentiment analysis(这是一个分类的问题，但是因为input是sequence，所以用RNN来处理)

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-39.png)
用RNN来作key term extraction。key term extraction意思就是说给machine看一个文章，machine要给出这篇文章有哪些关键词汇。那如果你今天能够收集到一些training data(一些document，这些document都有label，哪些词汇是对应的，那就可以直接train一个RNN)，那这个RNN吧document当做input，通过Embedding layer，然后用RNN把这个document读过一次，然后把出现在最后一个时间点的output拿过来做attention，你可以把这样的information抽出来再丢到feedforward neural得到最后的output


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-40.png)
那它也可以是多对多的，比如说当你的input和output都是sequence，但是output sequence比input sequence短的时候，RNN可以处理这个问题。咋样的任务是input sequence长，output sequence短呢。比如说，语音辨识就是这样的任务。在语音辨识这个任务里面input是vector sequence(说一句话，这句话就是一段声音讯号)。我们一般处理声音讯号的方式，在这个声音讯号里面，每隔一小段时间，就把它用vector来表示。这个一小段时间是很短的(比如说，0.01秒)。那output sequence是character sequence。

如果你是原来的RNN(slot filling的那个RNN)，你把这一串input丢进去，它充其量只能做到说，告诉你每一个vector对应到哪一个character。加入说中文的语音辨识的话，那你的output target理论上就是这个世界上所有可能中文的词汇，常用的可能是八千个，那你RNNclassifier的数目可能就是八千个。虽然很大，但也是没有办法做的。但是充其量只能做到说：每一个vector属于一个character。每一个input对应的时间间隔是很小的(0.01秒)，所以通常是好多个vector对应到同一个character。所以你的辨识结果为“好好好棒棒棒棒棒”。你会说：这不是语音辨识的结果呀，肿么办，有一招叫做“trimming”(把重复的东西拿掉)，就变成“好棒”。这这样会有一个严重的问题，因为它没有辨识“好棒棒”


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-41.png)
需要把“好棒”跟“好棒棒”分开来，肿么办，我们有一招叫做“CTC”(这招很神妙)，它说：我们在output时候，我们不只是output所有中文的character，我们还有output一个符号，叫做"null""(没有任何东西)。所以我今天input一段acoustic feature sequence,它的output是“好 null null 棒 null null null null”，然后我就把“null”的部分拿掉，它就变成“好棒”。如果我们输入另外一个sequence，它的output是“好 null null 棒 null 棒 null null”，然后把“null”拿掉，所以它的output就是“好棒棒”。这样就可以解决这样的问题了。


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-42.png)
那在训练neural咋样做呢(CTC咋样做训练呢)。CTC在做training的时候，你手上的train data就会告诉你说，这一串acoustic features对应到这一串character sequence，但它不会告诉你说“好”是对应第几个character 到第几个character。这该肿么办呢，穷举所有可能的alignments。简单来说就是，我们不知道“好”对应到那几个character，“棒”对应到哪几个character。假设我们所有的状况都是可能的。可能第一个是“好 null 棒 null null null”，可能是“好 null null 棒 null null”，也可能是“好 null null null 棒 null”。我们不知道哪个是对的，那假设全部都是对的。在train的时候，全部都当做是正确的，然后一起train。穷举所有的可能，那可能性太多了，有没有巧妙的算法可以解决这个问题呢？那今天我们就细讲这个问题。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-43.png)
以下是在文献CTC上得到的结果。在做英文辨识的时候，你的RNN output target 就是character(英文的字母+空白)。直接output字母，然后如果字和字之间有boundary，就自动有空白。

假设有一个例子，第一个frame是output h，第二个frame是output null，第三个frame是output null，第四个frame是output I等等。如果你看到output是这样子话，那最后把“null”的地方拿掉，那这句话的辨识结果就是“HIS FRIEND'S”。你不需要告诉machine说："HIS"是一个词汇，“FRIEND's”是一个词汇,machine通过train data会自己学到这件事情。那传说，Google的语音辨识系统已经全面换成CTC来做语音辨识。如果你用CTC来做语音辨识的话，就算是有某一个词汇(比如是：英文中人名，地名)在train data中从来没有出现过，machine也是有机会把它辨识出来。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-44.png)
另外一个神奇RNN的应用叫做sequence to sequence learning，在sequence to sequence learning里面,RNN的input跟output都是sequence(但是两者的长度是不一样的)。刚在在CTC时，input比较长，output比较短。在这边我们要考虑的是不确定input跟output谁比较长谁比较短。

比如说，我们现在做machine translation，input英文word sequence把它翻译成中文的correct sequence。那我们并不知道说，英文跟中文谁比较长谁比较短(有可能是output比较长，output比较短)。所以改肿么办呢？

现在假如说input machine learning ，然后用RNN读过去，然后在最后一个时间点，这个memory里面就存了所有input sequence的information。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-45.png)
然后接下来，你让machine 吐一个character(机)，然后就让它output下一个character，把之前的output出来的character当做input，再把memory里面的值读进来，它就会output “器”。那这个“机”这个地方呢，有很多支支节节的技巧，还有很多不同的变形。在下一个时间input “器”，output“学”，然后output“习”，然后一直output下去


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-46.png)
这就让我想到推文接龙，有一个人推超，下一个人推人，然后推正，然后后面一直推推，等你推好几个月，都不会停下来。你要咋样让它停下来呢？推出一个“断”，就停下来了。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-47.png)
咋样要阻止让它产生词汇呢？你要多加一个symnol “断”，所以现在manchine不只是output说可能character，它还有一个可能output 叫做“断”。所以今天“习”后面是“===”(断)的话，就停下来了。你可能会说这个东西train的起来吗，这是train的起来的。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-48.png)
这篇的papre是这样做的，sequence to sequence learning我们原来是input 某种语言的文字翻译成另外一种语言的文字(假设做翻译的话)。那我们有没有可能直接input某种语言的声音讯号，output另外一种语言的文字呢？我们完全不做语音辨识。比如说你要把英文翻译成中文，你就收集一大堆英文的句子，看看它对应的中文翻译。你完全不要做语音辨识，直接把英文的声音讯号丢到这个model里面去，看它能不能output正确的中文。这一招居然是行得通的。假设你今天要把台语转成英文，但是台语的语音辨识系统不好做，因为台语根本就没有standard文字系统，所以这项技术可以成功的话，未来你在训练台语/英文语音辨识系统的时候，你只需要收集台语的声音讯号跟它的英文翻译就可以刻了。你就不需要台语语音辨识的结果，你也不需要知道台语的文字，也可以做这件事。


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-49.png)
利用sequence to sequence的技术，甚至可以做到Beyond Sequence。这个技术也被用到syntactic parsing。synthetic parsing这个意思就是说，让machine看一个句子，它要得到这个句子的结构树，咋样让machine得到这样的结构呢？，过去你可能要用structure learning的技术能够解这个问题。但是现在有了 sequence to sequence learning的技术以后，你只要把这个树状图描述成一个sequence(具体看图中 john has a dog)。所以今天是sequence to sequence learning 的话，你就直接learn 一个sequence to sequence model。它的output直接就是syntactic parsing。这个是可以train的起来的，非常的surprised

你可能想说machine它今天output的sequence不符合文法结构呢(记得加左括号，忘记加右括号)，神奇的地方就是LSTM不会忘记右括号(这里没看懂)。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-50.png)
那我们要将一个document表示成一个vector的话，往往会用bag-of-word的方法，用这个方法的时候，往往会忽略掉 word order information。举例来说，有一个word sequence是“white blood cells destroying an infection”，另外一个word sequence是：“an infection destroying white blood cells”，这两句话的意思完全是相反的。但是我们用bag-of-word的方法来描述的话，他们的bag-of-word完全是一样的。它们里面有完全一摸一样的六个词汇，因为词汇的order是不一样的，所以他们的意思一个变成positive，一个变成negative，他们的意思是很不一样的。

那我们可以用sequence to sequence Auto-encoder这种做法来考虑word sequence order的情况下，把一个document变成一个vector

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-51.png)
input一个word sequence，通过Recurrent Neural  Network变成一个in value vector，然后把这个vlaue vector当做Encode的输入，然后让这个Encode，找回一模一样的句子。如果今天Recurrent Neural Network可以做到这件事情的话，那Encode这个vector就代表这个input sequence里面重要的information。你在trian Sequence-to-sequence Auto-encoder的时候，不需要label data，你只需要收集大量的文章，然后直接train下去就好了。


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-52.png)
这个结构甚至可以是high rect,你可以每一个句子都先得到一个vecto(mary was hungry得到一个vector，she didn't find any food得到一个vector)，然后把这些vector加起来，然后变成一个整个 document high label vector，在让这整个vector去产生一串sentence vector，在根据每一个sentence vector再去解回word sequence。这是一个四层的LSTM(从word 变成sentence sequence ，变成document lable，再解回sentence sequence，再解回word sequence)


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-53.png)
这个也可以用到语音上，在语音上，它可以把一段audio segments变成一个fixed length vector。比如说，左边有一段声音讯号，长长短短都不一样，那你把他们变成vector的话，可能dog跟dogs比较接近，never和ever比较接近。我称之为audio auto vector。一般的auto vector它是把word变成vector，这个是把一段声音讯号变成一个vector。


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-54.png)
那这个东西有什么用呢？它可以做很多的事。比如说，我们可以拿来做语音的搜寻。什么是语音的搜寻呢？你有一个声音的data base(比如说，上课的录音，然后你说一句话，比如说，你今天要找跟美国有关的东西，你就说美国，不需要做语音辨识，直接比对声音讯号的相似度，machine 就可以从data base里面把提到的部分找出来)


那这个肿么做呢？你就先把一个audio data base，把这个data base做segregation切成一段一段的。然后每一个段用刚才讲的audio segment to vector这个技术，把他们通通变成vector。然后现再输入一个spoken query，可以通过audio segment to vector技术也变成vector，接下来计算他们的相似程度。然后就得到搜寻的结果


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-55.png)
如何把一个audio segment变成一个vector呢？把audio segment抽成acoustic features，然后把它丢到Recurrent neural network里面去，那这个recurrent neural network它的角色就是Encode，那这个recurrent neural network读过acoustic features之后，最后一个时间点它存在memory里面的值就代表了input声音讯号它的information。它存到memory里面的值是一个vector。这个其实就是我们要拿来表示整段声音讯号的vector。


但是只要RNN Encode我没有办法去train，同时你还要train一个RNN Decode，Decode的作用就是，它把Encode存到memory里面的值，拿进来当做input，然后产生一个acoustic features sequence。然后希望这个$y_1$跟$x_1$越接近越好。然后再根据$y_1$产生$y_2$，以此类推。今天训练的target$y_1,y_2,y_3,y_4$跟$x_1,x_2,x_3,x_4$越接近越好。那在训练的时候，RNN Encode跟RNN Decode是一起train的

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-56.png)
我们在实验上得到一些有趣的结果，图上的每个点其实都是一段声音讯号，你把声音讯号用刚才讲的
Sequence-to-sequence Auto-encoder技术变成平面上一个vector。发现说：fear这个位置在左上角，near的位置在右下角，他们中间这样的关系(fame在左上角，name在右下角)。你会发现说：把fear的开头f换成n，跟fame的开头f换成n，它们的word vector的变化方向是一样的。现在还没有把语义加进去。


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-57.png)
现在有一个demo，这个demo是用Sequence-to-sequence Auto-encoder来训练一个chat-bot(聊天机器人)。咋样用sequence to sequence learning来train chat-bot呢？你就收集很多的对话，比如说电影的台词，在电影中有一个台词是“How are you”，另外一个人接“I am fine”。那就告诉machine说这个sequence to sequence learning当它input是“How are you”的时候，这个model的output就要是“I am fine”。你可以收集到这种data，然后就让machine去 train。这里我们就收集了四万句和美国总统辩论的句子，然后让machine去学这个sequence to sequence这个model。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-58.png)

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-59.png)
现在除了RNN以外，还有另外一种有用到memory的network，叫做Attention-based Model，这个可以想成是RNN的进阶的版本。

那我们知道说，人的大脑有非常强的记忆力，所以你可以记得非常非常多的东西。比如说，你现在记得早餐吃了什么，同时记得10年前夏天发生的事，同时记得在这几门课中学到的东西。那当然有人问你说什么是deep learning的时候，那你的脑中会去提取重要的information，然后再把这些information组织起来，产生答案。但是你的脑中会自动忽略那些无关的事情，比如说，10年前夏天发生的事情等等。



![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-60.png)
那其实machine也可以做到类似的事情，machine1也可以有很大的记忆的容量。它可以有很大的data base，在这个data base里面，每一个vector就代表了某种information被存在machine的记忆里面。

当你输入一个input的时候，这个input会被丢进一个中央处理器，这个中央处理器可能是一个DNN/RNN，那这个中央处理器会操控一个Reading Head 
Controller，这个Reading Head Controller会去决定这个readin head放的位置。machine再从这个reading head 的位置去读取information，然后产生最后的output

![im![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-61.png)
这个model还有一个2.0的版本，它会去操控writing head controller。这个writing head controller会去决定writing head 放的位置。然后machine会去把它的information通过这个writing head写进它的data base里面。所以，它不仅有读的功能，还可以discover出来的东西写入它的memory里面去。这个就是大名鼎鼎的Neural Turing Machine


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-62.png)
Attention-based Model 常常用在Reading Comprehension里面。所谓的Reading Comprehension就是让machine读一堆document，然后把这些document里面的内容(每一句话)变成一个vector。每一个vector就代表了每一句话的语义。比如你现在想问machine一个问题，然后这个问题被丢进中央处理器里面，那这个中央处理器去控制而来一个reading head controller，去决定说现在在这个data base里面哪些句子是跟中央处理器有关的。假设machine发现这个句子是跟现在的问题是有关的，就把reading head放到这个地方，把information 读到中央处理器中。读取information这个过程可以是重复数次,也就是说machine并不会从一个地方读取information，它先从这里读取information以后，它还可以换一个位置读取information。它把所有读到的information calculate起来，最后给你一个最终的答案。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-63.png)

读过这五个句子，然后说：waht color is Grey?，得到正确的答案，yes。那你可以从machine1 attention的位置(也就是reading head 的位置)看出machine的思路。图中蓝色代表了machine reading head 的位置，Hop1，Hop2，Hop3代表的是时间，在第一个时间点，machine先把它的reading head放在“greg is a frog”，把这个information提取出来。接下来提取“brian is a frog” information ，再提取“brian is yellow”information。最后它得到结论说：greg 的颜色是yellow。这些事情是machine自动learning出来的。也就是machine attention在哪个位置，这些通过neural network学到该肿么做，并不是去写程序，你要先看这个句子，在看这个句子。这是machine自动去决定的。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-64.png)
也可以做Visual Question Answering，让machine看一张图，问它这是什么，如果它可以正确回答说：这是香蕉，这就非常厉害了。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-65.png)
这个Visual Question Answering该肿么做呢？先让machine看一张图，然后通过CNN你可以把这张图的一小块region用一小块的vector来表示。接下里，输入一个query，这个query被丢到中央处理器中，这个中央处理器去操控这个reading head controller，这个reading head controller决定读取的位置(是跟现在输入的问题是有，这个读取的process可能要好几个步骤，machin，最后得到答案。


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-66.png)
那可以做语音的Question Answering 。比如说：在语音处理实验上我们让machine做TOEFL Listening Comprehension Test 。让machine听一段声音，然后问它问题，从四个选项里面，machine选择出正确的选项。那machine做的事情是跟人类考生做的事情是一摸一样的。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-67.png)
那用的Model Architecture跟我们刚才看到的其实大同小异。你让machine先读一个question，然后把question做语义的分析得到question的语义，声音的部分是让语音辨识先转成文字，在把这些文字做语音的分析，得到这段文字的语义。那machine了解question的语义然后就可以做attention，决定在audio story里面哪些部分是回答问题有关的。这就像画重点一样，machine画的重点就是答案，它也可以回头去修正它产生的答案。经过几个process以后，machine最后得到的答案跟其他几个选项计算相似度，然后看哪一个想项的相似度最高，它就选那一个选项。那整个大的test就是一个neural network。


![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-68.png)
这些是一些实验结果，这个实验结果是：random 正确率是25 percent。有两个方法要比25 percent要强的。

这五个方法都是naive的方法，完全不管文章的内容，直接看问题跟选项就猜答案。我们发现说，如果你选最短的那个选项，你就会得到35 percent的正确率。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-69.png)

memory network可以得到39.2 percent正确率，如果用我们刚才讲的那个model的话，可以做到48.8 percent正确率。

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-70.png)
**deep learning 个structured learning中间有什么样的关系呢。**

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-71.png)
使用RNN跟structure learning的技术有什么不同呢？首先假如我们用的是unidirectional RNN/LSTM，当你在  decision的时候，你只看了sentence的一半，而你是用structure learning的话，你考虑的是整个句子。从这个结果来看，HMMM，SVM等还是占到一些优势的。但是这个优势并不是很明显，因为RNN和LSTM他们可以做Bidirectional ，所以他们也可以考虑一整个句子的information

在HMM/SVM里面，你可以explicitly的考虑label之间的关系


如果是LSTM/RNN，你的cost function跟你实际上要考虑的error往往是没有关系的，当你做RNN/LSTM的时候，考虑的cost是每一个时间点的cross entropy(每一个时间的RNN的output cross entropy)，它跟你的error不见得是直接相关的。但是你用structure learning的话，structure learning 的cost会影响你的error，从这个角度来看，structure learning也是有一些优势的。最重要的是，RNN/LSTM可以是deep，HMMM,SVM等它们其实也可以是deep，但是它们要想拿来做deep learning 是比较困难的。在我们上一堂课讲的内容里面。它们都是linear

![image](http://ppryt2uuf.bkt.clouddn.com/chapter10-72.png)
deep learning和structure learning结合起来。input features 先通过RNN/LSTM，然后RNN/LSTM的output再做为HMM/svm的input。这个就同时享有deep learning的好处，也可以有structure learning的好处。


在语音上，我们常常把deep learning 和structure learning 合起来(CNN/LSTM/DNN + HMM)

在HMM里面，必须要去计算


# 参考
[有道云笔记原文](http://note.youdao.com/noteshare?id=feeb61c1ab765f1eaa8f2941fd3691ad&sub=E382D4AB086949EFA8F0447B2F9A6035)