# 李宏毅机器学习笔记(leeml-book)
李宏毅老师的[机器学习视频](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML17.html)是机器学习领域经典的中文视频之一，也被称为中文世界中最好的机器学习视频。李老师以幽默风趣的上课风格让很多晦涩难懂的机器学习理论变得轻松易懂，并且老师会通过很多有趣的例子结合机器学习理论在课堂上展现出来，并且逐步推导深奥的理论知识。比如老师会经常用宝可梦来结合很多机器学习算法。对于想入门机器学习又想看中文讲解的人来说绝对是非常推荐的。学有余力的同学也可以看一下[李宏毅机器学习2019](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML19.html)(大部分是一样的，只有小部分更新)


## 使用说明
这个笔记是根据李宏毅老师机器学习视频的一个辅助资料，本笔记基本上完全复刻李老师课堂上讲的所有内容，并加入了一些和相关的学习补充资料和参考资料，结合这些资料一起学习，相信你会对机器学习有更加深刻的理解。

### 在线阅读地址
在线阅读地址：https://datawhalechina.github.io/Leeml-Book/

# 目录
- [P1 0-1:Introduction of Machine Learning](chapter1/chapter1.md)
- [P2 0-2:Why we need to learn machine learning?](chapter2/chapter2.md)
- [P3 1-1:Regression - Case Study](chapter3/chapter3.md)
- [P4 1-2:Regression - Demo - YouTube (HD)](chapter4/chapter4.md)
- [P5 2:Where does the error come from?](chapter5/chapter5.md)
- [P6 3-1:Gradient Descent](chapter6/chapter6.md)
- [P7 3-2:Gradient Descent (Demo by AOE)](chapter7/chapter7.md)
- [P8 3-3:Gradient Descent (Demo by Minecraft)](chapter8/chapter8.md)
- [P9 HW1 – PM2.5 Prediction](chapter9/chapter9.md)
- [P10 4:Classification](chapter10/chapter10.md)
- [P11 5:Logistic Regression](chapter11/chapter11.md)
- [P12 HW2 - Winner or Loser](chapter12/chapter12.md)
- [P13 6:Brief Introduction of Deep Learning](chapter13/chapter13.md)
- [P14 7:Backpropagation](chapter14/chapter14.md)
- [P15 8-1:"Hello world" of deep learning](chapter15/chapter15.md)
- [P16 8-2:Keras 2.0](chapter16/chapter16.md)
- [P17 8-3:Keras Demo](chapter17/chapter17.md)
- [P18 9-1:Tips for Training DNN](chapter18/chapter18.md)
- [P19 9-2:Keras Demo 2](chapter19/chapter19.md)
- [P20 9-3:Fizz Buzz in Tensorflow (sequel)](chapter20/chapter20.md)
- [P21 10:Convolutional Neural Network](chapter21/chapter21.md)
- [P22 11:Why Deep?](chapter22/chapter22.md)
- [P23 12:Semi-supervised](chapter23/chapter23.md)
- [P24 13:Unsupervised Learning - Linear Methods](chapter24/chapter24.md)
- [P25 14:Unsupervised Learning - Word Embedding](chapter25/chapter25.md)
- [P26 15:Unsupervised Learning - Neighbor Embedding](chapter26/chapter26.md)
- [P27 16:Unsupervised Learning - Auto-encoder](chapter27/chapter27.md)
- [P28 17:Unsupervised Learning - Deep Generative Model (Part I)](chapter28/chapter28.md)
- [P29 18:Unsupervised Learning - Deep Generative Model (Part II)](chapter29/chapter29.md)
- [P30 19:Transfer Learning](chapter30/chapter30.md)
- [P31 20:Support Vector Machine (SVM)](chapter31/chapter31.md)
- [P32 21:Structured Learning - Introduction](chapter32/chapter32.md)
- [P33 22:Structured Learning - Linear Model](chapter33/chapter33.md)
- [P34 23:Structured Learning - Structured SVM](chapter34/chapter34.md)
- [P35 24:Structured Learning - Sequence Labelin](chapter35/chapter35.md)
- [P36 25:Recurrent Neural Network (Part I)](chapter36/chapter36.md)
- [P37 26:Recurrent Neural Network (Part II)](chapter37/chapter37.md)
- [P38 27:Ensemble](chapter38/chapter38.md)
- [P39 28:Deep Reinforcement Learning - Scratching the surface](chapter39/chapter39.md)
- [P40 The Next Step for Machine Learning](chapter40/chapter40.md)


## 视频观看地址
bilibili地址：[李宏毅机器学习(2017)](https://www.bilibili.com/video/av10590361/)
网易云课堂地址：[台大李宏毅机器学习中文课程](https://study.163.com/course/introduction/1208946807.htm)

#  协作规范

### 文档书写规范：
文档采用Markdown语法编写，数学公式采用LaTeX语法编写，数学符号和视频里完全一致，文中所用到的图片均来自[李宏毅机器学习课程主页](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML17.html)
，脑图则推荐使用[百度脑图](http://naotu.baidu.com)

|          | 格式     | 参考资料                                                     |
| :------: | :------- | :----------------------------------------------------------- |
| 文档 | Markdown | 1. CSDN Markdown 使用教程 http://t.cn/E4699cO<br>2. 简书 Markdown 使用教程 https://www.jianshu.com/p/q81RER<br>3. 编辑软件推荐 Typora https://typora.io/ |
| 数学公式 | LaTeX    | 1. CSDN Latex语法编写数学公式 http://t.cn/E469pdI<br>2.Latex 在线编辑工具 http://latex.codecogs.com/eqneditor/editor.php |


### 目录结构规范：

```
leeML-book
├─docs
|  ├─chapter1  # 第1章(对应在线网站的 P1 Introduction of Machine Learning)
|  |  ├─res  # 资源文件夹（图片、资料）
|  |  |  └─chapter1-1.png
|  |  ├─chapter1.md
|  ├─chapter2
...
|  ├─AdditionalReferences（所有补充资料）
|  |  ├─DecisionTree  
|  |  |  └─Entropy.md 
```


### 公式全解文档规范：
```
## 复现笔记在原则上是记录李老师课堂的话，对于比较繁琐的公式推导或者概念理解可以自己总结写上去
## 公式格式与PPT上保持一致
## 总结每篇笔记的脑图作为大纲,并选择百度脑图外观中的鱼骨图
## 每张图片下面都空一行(为了在电脑端显示格式不乱)
## 笔记开头都先放上脑图作为笔记内容概要(脑图命名chapterX-0.png，如有多张可按chapterX-0_1.png等等来命名)
## 每节笔记标题用一级标题，每节小标题用二级标题(目录只能识别一级和二级标题)
## 文中所有图片均放在同级的res文件夹下并按照chapterX-X.png格式命名(第一个X指章节编号，第二个X是指图片编号(从1开始))
## 保证文档语言的书面化和清晰的逻辑

```
### 修正记录：
版本|时间|作者|文档信息
---|:--:|:--:|:--|---
v1.0|2019.06.28|[@DatawhaleXiuyuan](https://github.com/DatawhaleXiuyuan)[@hahlw](https://github.com/hahlw)[@Heitao5200](https://github.com/Heitao5200)[@ImayKing](https://github.com/Imay-King)[@spareribs](https://github.com/spareribs)|建立初始仓库
v1.1|内容|内容|内容
v1.2|内容|内容|内容




样例参见[chapter38](https://github.com/datawhalechina/Leeml-Book/tree/master/docs/chapter38)

# 主要贡献者（按首字母排名）

- [@DatawhaleXiuyuan](https://github.com/DatawhaleXiuyuan)
- [@hahlw](https://github.com/hahlw)
- [@Heitao5200](https://github.com/Heitao5200)
- [@ImayKing](https://github.com/Imay-King)
- [@spareribs](https://github.com/spareribs)

# 关注我们

<div align=center><img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "250" height = "270" alt="Datawhale，一个专注于AI领域的学习圈子。初衷是for the learner，和学习者一起成长。目前加入学习社群的人数已经数千人，组织了机器学习，深度学习，数据分析，数据挖掘，爬虫，编程，统计学，Mysql，数据竞赛等多个领域的内容学习，微信搜索公众号Datawhale可以加入我们。"></div>


