
![image](6EE26AB477B74E32BC6A3A86F37329B9)
support vector machine它有两个特色，第一个是用了hinge loss，另外一个厉害的地方是kernal method




![image](333C368B1A3B45FBA226243652CDB80A)

Binary classification在理想下我们期待说：一个machine的solution往往就有三个step。在Binary classification里面第一个step是：定义一个function g(x)，这个g(x)里面有function f(x)，当function 大于0的时候，它的output就是+1代表一个class，当f(x)小于0的时候，output就是-1代表另外一个class。那现在的training data是supervised(每一笔data都有一个label`$\hat{y}^n$`)，现在假设`$\hat{y}^n$`用+1、-1分别来代表两个不同的class。


最理想的loss function就是写成下面这个样子，当`$g(x^n)$`跟`$\hat{y}^n$`不一样的时候就写成图中这个样子，一样的时候就没有loss。所以一个很理想的loss function是summation over我们所有的train data，然后对每一笔train data都代进去`$g(x)$`看它的output是多少(有可能output  +1，也有可能output-1)，接下来看跟`$\hat{y}^n$`一样的还是不一样的。不一样的就得到了一个loss，一样的就得到0。所以我们的loss就变成g(x)在train 上犯了几次错误，我们希望它犯的错误是越小越好。


但是在今天这个task里面，在第三步找一个好的function是有点困难的，因为你的loss是不可微分的，没有办法用gradient descent来解它。那肿么办呢？

我们把这一项用另外一个loss来表示(直接minimize这一项做不到，所以我们minimize另一项)，至于loss function长什么样子，可以自己来定义。



![image](4C3A73BD7B824A7F80D92F3366DCA183)

这个图像的横轴`$\hat{y}f(x)$`(`$\hat{y}$`可以是+1，也可以是-1)，如果希望`$\hat{y}^n$`是+1的话，f(x)越postive越好，`$\hat{y}^n$`是-1的话，f(x)越负越好，所以整体说来你会希望`$\hat{y}^n$`跟f(x)两个值相乘以后越大越好(同号)。

如果纵轴是loss的话，越往右`$\hat{y}f(x)$`越好，loss越小。我们刚才讲了理想的状况是：假设今天`$\hat{y}^n$`与f(x)是反向的，它们相乘以后你得到的值是负数的话，那你得到的loss就是1。反之它们是同向的，loss就是0，这只是理想的状况，这件事情是没有办法进行微分的。

![image](61C16BF6D70F42CB94AD626885163DFE)
那我们先用L把δ来取代，纵轴就是L的值，可以选择各式各样的function来当做L这个function。举例来说：现在的loss定法是：当`$\hat{y}^n=1$`时，f(x)跟1越接近越好；`$\hat{y}^n=-1$`时，f(x)跟-1越接近越好
，这个是square los，可以写成`$(\hat{y}^nf(x^n)-1)^2$` 。这是为什么呢？如果`$\hat{y}^n$`等于1的话，这个function就变为`$(f(x^n)-1)^2$`，如果`$\hat{y}^n$`等于-1的话，这个function就变为`$(-f(x^n)-1)^2$`，`$(f(x^n)+1)^2$`，这个图形是不合理的(如图)。我们一开始讲Binary classification的时候，讲过你用square loss是不合理的，从这个图上你更可以看出它的不合理性，因为我们不希望说`$\hat{y}f(x)$`很大的时候，居然有一个很大的loss。


![image](4980E1964917491B96C941635DC864C8)
另外一个是sigmoid+square loss(sigmoid function用一个`$\sigma$`来表示)，我们希望`$\hat{y}^n=1$`的时候，`$\sigma(f(x))$`去接近于1，`$\hat{y}^n=-1$`的时候，`$\sigma(f(x))$`去接近于0，这个式子可以写成这个样子`$(\sigma(\hat{y}^nf(x))-1)$`。这样写是为什么呢？如果你把`$\hat{y}^n=1$`的时候，很直觉就是`$\sigma(f(x)-1)$`，`$\hat{y}^n=-1$`的时候，就是`$\sigma(-f(x)-1)$`，画出来就是蓝色这条线。我们之前说在做logistic regression的时候，你不会用square loss 当做你的loss，因为它的performance不好。


![image](FBB9C31AB14B486C94E47C378F5B089D)
cross entropy之前就有讲过：`$\sigma(f(x))$`代表了一个distribution，ground truth是另一个distribution。这样的distribution之间的cross entropy就是你要minimize的loss。这个function可以写成`$ln(1+exp(-\hat{y}^n)(x)))$`，这个式子是合理的。你想一下：当`$\hat{y}^nf(x)$`趋近于正无穷大的时候，`$exp(\hat{y}^nf(x))$`为0，就变为`$ln(0)=0$`，`$\hat{y}^nf(x)$`趋近于负无穷大的时候，`$1+(exp(\hat{y}^nf(x)))$`也很大，再去ln之后它还是会很大。所以是绿色的这条线


可以比较蓝色和绿色这两条曲线，你就可以了解说为什么会选择cross entropy而不是选择square error来当做loss function(logistic regression的时候)。我们今天`$\hat{y}f(x)$`从-2移到-1的时候，如果是sigmoid+square loss的时候变化很小，如果是sigmoid+cross entropy的时候变化就是非常大。所以对sigmoid function来说，它在这种极端的case，你的值在negative时应该有很大的gradient，应该赶快调整你的值。但是实际上不是如此，当`$\hat{y}f(x)$`negative的时候，你调整你的值对total loss影响不大，所以就会变成说：对这个sigmoid+square loss来说，它就算调整了negative的值，它也没有办法得到太多的回报，所以它就不想调整那些negative的值。对sigmoid+cross entropy来说，它的努力是可以得到回报的，所以它就会很乐意把原来很negative的值把它往正的地方推。所以我们用cross entropy的时候，你会比square error更容易training。



![image](F6882987B62C445E972A01046C998453)
hinge loss它是写成一个很特别的式子：写成`$max(0,1-\hat{y}f(x))$`，如果`$\hat{y}^n$`等于1，那loss function就是`$max(0,1-\hat{y}f(x))$`。那什么样的状况下会有一个zero loss呢，只要`$1-f(x)<0$`就行了(f(x)>1)，这个loss的值就会是0。如果`$\hat{y}^n$`等于-1的时候，loss function就是max(0,1+f(x))，那要loss等于0，就要1+f(x)<0，也就是f(x)<-1。所以你用hinge loss做training的时候，什么时候machine觉得做到loss=0，最完美的情况呢？如果对于postive example来说， `$f(x)>1$`的时候就是完美的case。对negative来说，`$f(x)《-1$`的时候就是完美的case。如果把hinge loss画出来的话就是紫色这条线。

hinge loss在右半段的时候，只要`$\hat{y}f(x)$`大于1的时候，你的loss就是0(非常好)，再更大也没有帮助了。若`$\hat{y}f(x)$`是postive example这还不够好，它会说你只得到正确的答案还不够好，你要比正确的答案还要好过一段距离，这个距离就是margin。也就是说：要`$\hat{y}f(x)$`还没有大于1的时候，还是有penalty的，促使`$\hat{y}f(x)$`>1。

你可能会困惑说：为什么这里是1呢(`$max(0,1-\hat{y}f(x))$`)，那这边如果是1的话，hinge loss才会是ideal loss的optimal，如果你用其他值的话就不会是optimal。所以hinge loss跟刚才的cross entropy一样也是我们ideal loss的optimal。所以我们也会期待说：minimize hinge loss可能会得到minimize ideal loss的效果。


如果我们要比较hinge loss跟cross entropy的话，你会发现说最大的不同来自于：它们对应已经做的好的example的态度。如果我们今天把`$\hat{y}f(x)$`从1移动到2，对cross entropy来说：你可以得到loss的下降，所以cross entropy想要“好还要更好”。如果采用hinge loss，它是一个“及格就好”的loss function，今天只要值大过margin的时候就结束了，不会想要做的更好。那在实战上cross entropy跟hinge loss有什么样的差别呢？在实战上差别可能没有那么显著，有时候会看到hinge loss会略胜过cross entropy，其实也没有赢那么多

![image](489ED513AA46429AB776DCDB8EE1B55F)
linear SVM是说：我们现在的function就是linear(`$f(x)=\sum_iw_ix_i+b$`)，我们可以这个看做是两个vector的inner product，我们把w跟b串起来的那个vector直接用vector来表示，这个是model的参数，是要通过train data要找出来的，x跟1串起来的vector当做是一个新的feature x，所以一个function就写成w的transpose乘以x，在SVM里面f(x)是这样的，然后你就可以说：f(x)大于0时候属于一个class，f(x)小于0的时候是属于另一个class。

SVM里面的loss function的特色就是：它采用了hinge loss这个loss，通常你还会加上一个regularization term。这个loss function是一个convex function，因为hinge loss是一个convex function，L2-Norm也是一个convex function，你所有的loss都是convex function，当把这些convex function叠加起来也是convex function。


如果是convex function的话做gradient descent就很简单，不管从哪个地方做iteration，最后找出来的结果都是一样的。你可能会说这个可能在某些点不可微(hinge loss在某些地方有菱角)，你把这些不可微的convex function堆叠起来就是如图，如图看起来不是每个地方都是不可微的，都是可微的。我们在前面讲deep learning 时有Relu activation function，他们表面看起来是不可微的，但是你都是可以用gradient descent去做。所以今天这个case也一样可以用gradient descent。


我们比较logistic regression跟linear SVM的差别，唯一的差别就是：我们咋样定义loss function。你用hinge loss就是linear SVM，你用cross entropy就是logistics regression。

这个function没有必要一定要是linear的，如果它是linear的话是有很多的特值。如果不是linear的也是ok的，也可以用gradient descent来train。所以SVM是有deep version





![image](EA765B5CC00341A0859DED3F2D8D4262)
SVM可以用gradient descent来train，我们现在的loss function是`l(f)=\sum _n(f(x^n),\hat{y}^n)$`，gradient descent很简单，你只要能够对它做微分就好了。我们只要能够对model某一个`$w_i$`做偏微分就可以了，`$w_i$`只跟`$f(x^n)$`有关，所以你要用链式求导，求导结果就是`$x^n$`的第i个dimension。

前面`l(f)=\sum _n(f(x^n),\hat{y}^n)$`对`$w_i$`做偏微分。这个是hinge loss function，这个hinge loss function两个range。它可以output在0这个case，可以output在`$1-\hat{y}^nf(x^n)$`这个case。那什么时候会output在哪个range呢？这个是depend现在model w是多少。也就是说：假如你现在的`$1-\hat{y}^nf(x^n)$`>0(`$\hat{y}^nf(x^n)$`<1)。当`$\hat{y}^nf(x^n)$`<1的时候，你的model作用在这个range。对`$f(x^n)$`做微分，得到的是就是`$-\hat{y^n}$`，另外一个做微分以后就是，所以微分值就是两个可能。

所以L对`$w_i$`做偏微分的值如图，summation over所以的train data，在看每一笔train data`$\hat{y^n}f(x^n)$`是不是小于1，如果小于1这一项的值就是1，结果为`$-\hat{y}^nx^n_i$`。

现在把一项写作为`$c^n(w)$`，接下来就可以update你的参数了，SVM可以用gradient descent来解的。


![image](D5188F94B53D4D909FE1B1006F031769)
你可能全会说：这跟平常看到的SVM不太一样，我现在就把hinge loss变成平常看到的SVM。

我们现在把这个hinge loss用一个natation `$ε^n$`来代替(`$\varepsilon ^n=max(0,1-y^n f(x^n))$`)，我们现在的目标就是minimize这个total loss。`$\varepsilon ^n=max(0,1-y^n f(x^n))$`有另外一个写法：我们要0跟`$1-\hat{y}^nf(x^n)$`取大的那个当做`$ε^n$`，所以`$ε^n$`它会大于0，也大于`$1-\hat{y}^nf(x^n)$` ，那我可以写成`$ε^n\geqslant 0$`，`$ε^n\geqslant1-\hat{y}^nf(x^n)$`。如果我们无视这个L(f)的话，这两个式子是不一样的。但是加了minimize L(f)以后，这两个红色框的式子就会变得一模一样了。

因为我们现在要去minimize L(f)，所以你要去选择一个最小的`$ε^n$`让你的L能够最小。虽然我们用constraint `$ε^n \geqslant 0$` `$1-y^n f(x^n))$`。 `$ε^n$`可以选择无穷大·，理论上是可以选择满足这个条件的任何值。但问题是我们现在要去做的事就是minimize L，当我们要去minimize L的时候，我们就要去想办法希望 `$ε^n$`越小越好。

当constraint `$ε^n \geqslant 0$` `$1-y^n f(x^n))$`，让它最小的办法就是让 `$ε^n$` 等于里面最大的那个。所以加上我们的目标：要minimize total loss的时候，上面这个红色框框的式子等于下面这个红色框框的式子。

这个就是你所熟悉的SVM，你所熟悉的SVM就是告诉我们说：`$\hat{y}^n$`乘以f(x)要是同号的，相乘以后要大于等于一个margin 1，有时候没有办法让`$\hat{y}^n$`大于1，那要肿么办呢？所以你要稍微把你的margin做稍微的放宽，让它减去一个`$ε^n$`，这个`$ε^n$`会放宽你的margin。所以这个`$ε^n$`叫做slack variable。这个slcak variable不能是负的，这个就不符合它的目的，如果`$ε^n$`是负的话，那你就是把`$ε^n$`变大了，所以这个`$ε^n$`有一个constraint是要大于等于0


把这些事情合起来以后，你有一个minimize的对象再加上一些constraint。这个formulation是一个Quadratic Programming Probelm



![image](B38DAA2AAE594999B5EE1913C47F3C21)

![image](8C158977393748F7AEC69EE0D707FD4B)
实际上我们找出来可以minimize loss function那个weight写作`$ w^\ast $`，它其实是我们data的linear combination(summation over所有的 train data `$x^n$`，然后对所有的`$x^n$`乘以一个`$\alpha^\ast _n$`)，也就是说：你找出来的model就是data point 的linear combination。

我们刚才用gradient descent minimize SVM，gradient descent的式子是这样的`$w_i=w_i-\eta \sum _nx^n(w_i)x^n_i$`，唯一不同的地方就是最后乘上的value(在update`$w_1$`的时候，乘以`$x^n_1$`，以此类推)。这意味着假设initialization 的时候你的w是一个zero vector，你每次update w的时候都是乘上data point 的linear combination。所以最后得到的solution，gradient descent解出的w就是w的linear combination。

`$c^n(w)$`是loss 对`$f(x^n)$`的偏微分，如果我们今天用的是hinge loss，有两个range。如果是作用于max=0的range，那这个就是0。所以当你用hinge loss的时候，你的这一项往往都是0(`$c^n(w)x^n$`)，也就是：不是所以的`$x^n$`都能加到`$w$`里面去。所以你最后解出来的`$w^\ast$`，它的linear combination 的weight可能是sparse(有很多data point对应的`$\alpha^\ast_n$`等于0)
，而那些`$\alpha^\ast_n$`不等于0的`$x^n$`就是support vectors



在data point里面不是所有的点都是被选择为support vector，其实是少数的点被选择为support vector，所以SVM相较于其它的方法是sparse。如果今天的loss function选择的是cross entropy，它就没有sparse这个特性。cross entropy loss function在每个地方微分都是不等于0的，所以你今天解出来的`$\alpha^{\ast}_n$`就不会是sparse，用hinge解出来的就是sparse。


今天那些不是support vector的data point，你把它从database中remove掉，它对最后的影响是一点影响都没有的。不像是其他的方式(logistics regression)，每一笔data对最后的结果都会造成影响。


![image](5E2E1D10999147EB86D3ECA16187424C)
把w写成data point的linear combination，最大的好处就是我们可以使用kernal method


首先我们知道w是data point的linear combination，我们可以把`$x^1$`到`$x^n$`排成一个matrix X，`$\alpha$`是一个vector(`$\alpha_1,\alpha_N$`)。这个matrix乘以这个vector，你就会得到`$X\alpha$`(w=`$X\alpha$`)。当我们知道w可以这样写以后，我们可以改一下function的样子。我们原来function是`$f(x)=w^Tx$`，因为w=`$X\alpha$`，所以f(x)=`$\alpha^TX^Tx$`(x是一个vector，X的transpose是一堆row叠在一起，`$\alpha$`是一个倒的vector)。x乘以matrix `$X^T$`结果是：第一个dimension是`$x^1$`跟x的inner product，以此类推。把这两个vector再做inner product的结果就是：你的f(x)等于summation over`$\alpha^n$`跟(`$x^nx$`)的inner product。你可能觉得database里每一个x都要算一个inner product会不会很费事呢？其实还好。假如你是用hinge loss，这个`$\alpha$`是sparse，所以你只需要考虑不等于0的就好了。

我们把`$x^n$`跟x这件事写成一个function K(`$x^n,x$`)(`$x^n$`跟x做inner product)，这个function叫做kernel functon。




![image](0851A8E40A394FF29E3D686F96A58B47)
我们已经知道step1就是：`$x^n$`跟x代进kernel function，乘以`$\alpha_n$`，再summation over以后的结果。对于step2、step3：我们今天maximize的对象是什么呢？我们不知道参数是`$\alpha_n$`，我们的问题就变成了，你要找一组最好的`$\alpha_n$`让total loss最小。这个最好的`$\alpha_n$`长什么样子呢，loss function就是summation over每一笔data的loss function，把`$f(x^n)$`带进来就是这一项。观察投影片上所有的式子，你会发现说：现在我们真的不需要真的说vector x 是多少，我们真正需要知道的其实只有x跟z之间的inner product的值。

我们今天只需要知道kernal `${x^n}'$` `$x^n$`的value是什么，我并不需要真的去知道`${x^n}'$` `$x^n$`的vector长什么样子(我只要算出这一项就结束了)，这一招叫做Kernal Trick。




![image](C547BF10279F4B0983F36638EDE6C523)
我们之前说过如果是linear model，它有很多的限制 。你可能要对input feature做一个feature transpose，它才能用linear model来处理。如果在neural network里面我们就用好几个hidden layer来作feature transpose 。假设我们现在有一笔data(二维)，那我们想要对它先进行feature transpose，在feature transpose上面再去apply SVM。假设feature transpose以后的结果是`$\phi (x)$`(考虑feature跟feature之间在`$x_1,x_2$`上的关系)。

那我要算kernal(x,z)的时候(想要算x跟z之间的kernal function)，也就是x，z做完feature transpose以后inner product的值，可以肿么做呢？最简单的方法当然就是：我把x跟z都带到这个feature transpose function里面，把它们变成新的feature。变成新的feature以后，就可以做inner product。算出来的结果如图。`$(x_1z_1+x_2z_2)^2$`可以写为`$\begin{bmatrix}
x_1\\ 
x_2
\end{bmatrix}*\begin{bmatrix}
z_1\\ 
z_2
\end{bmatrix}$`。也就是`$\begin{bmatrix}
x_1\\ 
x_2
\end{bmatrix}$`跟`$\begin{bmatrix}
z_1\\ 
z_2
\end{bmatrix}$`这两个vector的inner product的平方。所以说：我们把x跟z做feature transpose以后，再做inner product以后。等同于原来在feature transpose 之前space上面先做inner product以后，再平方。

这一招这可以为我们带来好处，因为有时候直接计算的结果，x跟z代进kernal function的output会比先做feature transpose，再做inner product还要更快速。





![image](EC6FFAAB9E864C83A2A2D28758449FFB)
举例来说：假设我们现在要做的事情是(x跟z是高维vector)，把x跟z投影到更高维的平面。这更高维的平面我们会考虑所有feature 两两之间的关系，你最起码得算`$k^2$`维。

那如果用kernal Trick1的话，你可以轻易的把`$\phi (x)$`跟`$\phi (z)$`的结果轻易的算出来。`$\phi (x)$`跟`$\phi (z)$`的inner product就是`$x$`跟`$z$`的inner product的平方。你直接把x跟z做inner product再平方，你只需要算k个elements相乘，再做平方就好了。但是你如果先project到high dimension再做inner product的话会比先做inner Product再取平方的运算量要大。





![image](4B957BF4AFC6455188BCCBDBDF77DD31)
还有更惊人的结果，做Radial Basis function kernal。Radial Basis function kernal意思是说：kernal(x,z)就等于x跟z的距离乘以`$-\frac{1}{2}$`在取exponential，这个就是在衡量x跟z之间的相似度(如果x跟z越像，kernal的值越大，越不像就是0)。




这个式子其实也可以写成两个high dimension vector做inner product后的结果，这两个vector其实dimension是有无穷多维的。所以你要本来把一个x project到无穷多维子再做inner product，你做不到，因为根本不知道无穷多维是什么样根本不知道。但是你直接算x跟z之间的距离，乘以`$-\frac{1}{2}$`,再去exponential，其实就等同于在无穷多维空间里面去做inner product。这个无穷多维长什么样子呢？


我们可以把这一项展开，变为`$C_xC_zexp(x*z)$`(`$C_x,C_z$`代替前面那两项)，exp(x*z)用泰勒展开来表示。如果我们今天把这每一项都拆开的话，你会得到：`$C_z,C_z$`你可以看成是两个vector的inner product。`$C_xC_z(x,z)$`可以看成是：把原来的x vector乘以`$C_x$`，原来的z vectorv乘以`$C_z$`，在做inner product以后的结果。`$C_xC_z\frac{1}{2}(x*z)^2$`：x跟z的inner product再平方其实可以看成，两个high  dimension的vector再inner product的结果。x跟z的平方可以看成两个high dimension vector做inner product的结果，这个high dimension要考虑两个dimension之间的关系。

我们把x有关的vector串起来(一个很长的vector)，把z也串起来。这边有无穷多项，所以串起来x跟z都要各自无穷多个vector，再做inner product，最后得到的结果就是kernal (x,z)。
所以当你使用Radial Basis function Kernal的时候，你就是在无穷多维的平面上去做事情，在无穷维做事情很容易overfitting。所以你用Radial Basis function Kernal要小心，你可能在train data上得到很好的performance，在testing data上得到很糟糕的performance。




![image](21E7BF863F194F73911A8B1F52C09C1E)
你也可以做sigmoid kernal，sigmoid kernal是说：x跟z做inner product，再做tanh。我们之前说过：当我们要把x当做testing的时候，代到f里面。其实去算x跟`$x^n$`的kernal function的output，在乘以`$\alpha^n$`。当我们用sigmoid kernal 的时候，你就是把所有x跟`$x^n$`做inner product，取tanh，再乘以`$\alpha$`。

如果我们今天用的sigmoid kernal，这个f(x)你就可以想成它其实就是只有一个hidden layer的neural network。为什么呢？你把x拿进来它跟所有的`x^n$`做inner product，再取tanh。对x做inner product这件事就好像是：你有一个neural ，它的weight就是每一笔data(`$x^1$`，依次类推)，再通过tanh得到output。然后再把它全部乘以`$\alpha_n$`，最后加和起来得到f(x)。

这就是一个neural network，只不过就是一个hidden layer。在这个neural network里面，weight就是每一笔data，neural的数目是：看你有几个support vector，你就有几个neural。




![image](C02F56D51CB34D35AB3968A41AB4132E)
有了这个kernal Trick以后，我们可以直接去设计这个kernal function，我们根本完全不用理会x跟z的feature长什么样子。应当有一个kernal function可以把x跟z代进去，可以给你一value，这个value代表了x跟z在某一个高维平面上面的inner product，你根本不需要去在意x跟z它们的vector长什么样子。

这一招什么时候会有用呢？假设你的x是structure data(sequence)，如果是sequence的话你其实不容易把sequence表示成一个vector。(假设每一个sequence的长度不一样，那你就不容易把这些不同的长度的sequence，用一个vector来描述它)。但是我们可以定它的kernal function，我们知道kernal function其实就是投影到高维以后的inner product，kernal function往往就是类似similarity的东西。所以今天如果你可以定义一个function，它是evaluate x跟z的similarity。不是所以的function都可以有，但是有一个Mercer's theory可以告诉你哪些function是可以的，所以你有办法check定出来的kernal function它背后有没有两个vector做inner product这件事情。


在语音上，假设你现在要做分类的对象是Audio Segment，每一段声音讯号用Audio sequence来描述它(每一段声音讯号长度都不一样，所以Audiosequence都不一样)。假设你现在做的task可能是：给你一段声音讯号，它要看说这段声音讯号语者的情绪，可能分成高兴



61:34




















