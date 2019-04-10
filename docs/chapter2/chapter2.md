# 课程介绍
上节课[《李宏毅·机器学习》读书笔记（一）Regression - Case Study](https://blog.csdn.net/q370835062/article/details/87487127)，主要介绍了回归算法的整个演算过程。在课程最后为了改善模型，不断提升模型的复杂度，但是效果反而变差了。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190219105321809.png)
本节课主要介绍其他改善模型的方法，并介绍交叉验证这种模型选择的方案。


@[toc]

# Error的来源
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190219124034245.png)
从上节课测试集数据来看，$Average\ Error$ 随着模型复杂增加呈指数上升趋势。更复杂的模型并不能给测试集带来更好的效果，而这些 $Error$ 的主要有两个来源，分别是 $bias$ 和 $variance$ 。

然而 $bias$ 和 $variance$ 是什么？可以查看 [机器学习中的Bias(偏差)，Error(误差)，和Variance(方差)有什么区别和联系？](https://www.zhihu.com/question/27068705)

# 估测
假设真实的模型为 $\hat f$ ， 如果我们知道 $\hat f$ 模型，那是最好不过了，但是 $\hat f$ 只有 Niamtic 公司才知道。
![](https://img-blog.csdnimg.cn/20190219124007921.png)
所以我们只能通过收集 Pokemon精灵 的数据，然后通过 step1~step3 训练得到我们的理想模型 $f^*$，$f^*$ 其实是  $\hat f$ 的一个预估。

![](https://img-blog.csdnimg.cn/20190219124541290.png)
这个过程就像打靶，$\hat f$ 就是我们的靶心，$f^*$ 就是我们投掷的结果。如上图所示，$\hat f$ 与  $f^*$ 之间蓝色部分的差距就是 $bias$ 和 $variance$ 导致的。

## 估测变量x的偏差（bias）和方差（variance）
我们先理解一下偏差和方差是怎样计算的呢？ [偏差(Bias)和方差(Variance)——机器学习中的模型选择](https://segmentfault.com/a/1190000016447144)

### 评估 x 的平均值
- 假设 $x$ 的平均值是  $\mu$，方差为 $\sigma^2$

评估平均值要怎么做呢？

- 首先拿到 $N$ 个样本点：$\{x^1,x^2,···,x^N\}$
- 计算平均值 $m$, 得到 $m=\frac{1}{N}\sum_n x^n \neq \mu$
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019021915162184.png)

但是如果计算很多组的 $m$ ，然后求 $m$ 的期望：

$$E[m]=E[\frac{1}{N}\sum x^n]=\frac{1}{N}\sum_nE[x^n]=\mu$$

这个估计呢是无偏估计（unbiased）。

然后 $m$ 分布对于 $\mu$ 的离散程度（方差）：
$$Var[m]=\frac{\sigma^2}{N}$$

这个取决于 $N$，下图看出 $N$ 越小越离散：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190219152829679.png)

### 估测变量 x 的方差（variance）
如何估算 $variance$ 呢？

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190219113020805.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190219113342137.png)

## 为什么会有很多的 $f^*$ ?
讨论系列02中的案例：这里假设是在平行宇宙中，抓了不同的神奇宝贝
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019021915362235.png)


用同一个model，在不同的训练集中找到的 $f^∗$ 就是不一样的
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019021915365091.png)


这就像在靶心上射击，进行了很多组（一组多次）。现在需要知道它的散布是怎样的，将100个宇宙中的model画出来

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190219153711886.png)

不同的数据集之前什么都有可能发生—||

### 考虑不同 model 的 variance


一次model的variance就比较小的，也就是是比较集中，离散程度较小。而5次model 的 variance就比较大，同理散布比较广，离散程度较大。

所以用比较简单的model，variance是比较小的（就像射击的时候每次的时候，每次射击的设置都集中在一个比较小的区域内）。如果用了复杂的model，variance就很大，散布比较开。

这也是因为简单的model受到不同训练集的影响是比较小的。

### 考虑不同 model的 bias
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190219153750674.png)

这里没办法知道真正的 $\hat{f}$，所以假设图中的那条黑色曲线为真正的 $\hat{f}$


结果可视化，一次平均的 $\bar{f}$ 没有5次的好，虽然5次的整体结果离散程度很高。



一次model的bias比较大，而复杂的5次model，bias就比较小。

直观的解释：简单的model函数集的space比较小，所以可能space里面就没有包含靶心，肯定射不中。而复杂的model函数集的space比较大，可能就包含的靶心，只是没有办法找到确切的靶心在哪，但足够多的，就可能得到真正的 f¯f¯。

### bias v.s. variance
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190219153946423.png)

将系列02中的误差拆分为bias何variance。简单model（左边）是bias比较大造成的error，这种情况叫做 Underfitting（欠拟合），而复杂model（右边）是variance过大造成的error，这种情况叫做Overfitting（过拟合）。

# 怎么判断？
## 分析
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190219154002531.png)

如果model没有很好的fit训练集，就是bias过大，也就是Underfitting
如果model很好的fit训练集，即再训练集上得到很小的error，但在测试集上得到大的error，这意味着model可能是variance比较大，就是Overfitting。
对于Underfitting和Overfitting，是用不同的方式来处理的

### bias大，Underfitting
此时应该重新设计model。因为之前的函数集里面可能根本没有包含f^f^。可以：

将更多的feature加进去，比如考虑高度重量，或者HP值等等。
或者考虑更多次幂、更复杂的model。
如果此时强行再收集更多的data去训练，这是没有什么帮助的，因为设计的函数集本身就不好，再找更多的训练集也不会更好。

### variance大，Overfitting
简单粗暴的方法：More data

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190219154145269.png)

但是很多时候不一定能做到收集更多的data。可以针对对问题的理解对数据集做调整（Regularization）。比如识别手写数字的时候，偏转角度的数据集不够，那就将正常的数据集左转15度，右转15度，类似这样的处理。

# 选择model
现在在bias和variance之间就需要一个权衡
想选择的model，可以平衡bias和variance产生的error，使得总error最小
但是下面这件事最好不要做：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190219154201678.png)

用训练集训练不同的model，然后在测试集上比较error，model3的error比较小，就认为model3好。但实际上这只是你手上的测试集，真正完整的测试集并没有。比如在已有的测试集上error是0.5，但有条件收集到更多的测试集后通常得到的error都是大于0.5的。

## Cross Validation（交叉验证）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190219154215197.png)

图中public的测试集是已有的，private是没有的，不知道的。Cross Validation 就是将训练集再分为两部分，一部分作为训练集，一部分作为验证集。用训练集训练model，然后再验证集上比较，确实出最好的model之后（比如model3），再用全部的训练集训练model3，然后再用public的测试集进行测试，此时一般得到的error都是大一些的。不过此时会比较想再回去调一下参数，调整model，让在public的测试集上更好，但不太推荐这样。（心里难受啊，大学数模的时候就回去调，来回痛苦折腾）

上述方法可能会担心将训练集拆分的时候分的效果比较差怎么办，可以用下面的方法。

## N-fold Cross Validation（N-折交叉验证）
将训练集分成N份，比如分成3份。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190219154226105.png)

比如在三份中训练结果Average Error是model1最好，再用全部训练集训练model1。（貌似数模也干过，当年都是莫名其妙的分，想想当年数模的时候都根本来不及看是为什么，就是一股脑上去做00oo00）
