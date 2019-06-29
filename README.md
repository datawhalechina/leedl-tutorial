# 李宏毅机器学习笔记(leeml-book)
李宏毅老师的[机器学习视频](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML17.html)是机器学习领域经典的中文视频之一，也被称为中文世界中最好的机器学习视频。李老师以幽默风趣的上课风格让很多晦涩难懂的机器学习理论变得轻松易懂，并且老师会通过很多有趣的例子结合机器学习理论在课堂上展现出来，并且逐步推导深奥的理论知识。比如老师会经常用宝可梦来结合很多机器学习算法。对于想入门机器学习又想看中文讲解的人来说绝对是非常推荐的。学有余力的同学也可以看一下[李宏毅机器学习2019](http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML19.html)(大部分是一样的，只有小部分更新)111


## 使用说明
这个笔记是根据李宏毅老师机器学习视频的一个辅助资料，本笔记基本上完全复刻李老师课堂上讲的所有内容，并加入了一些和相关的学习补充资料和参考资料，结合这些资料一起学习，相信你会对机器学习有更加深刻的理解。

### 在线阅读地址
在线阅读地址：https://datawhalechina.github.io/Leeml-Book/

# 目录
- [学习大纲](index.md)
- [第0课 简介](chapter0/chapter0.md)
- [第1课 回归:案例研究](https://github.com/datawhalechina/Leeml-Book/blob/master/docs/chapter2)
- [第2课 误差分析](https://github.com/datawhalechina/Leeml-Book/tree/master/docs/chapter4)
- [第3课 梯度下降](https://github.com/datawhalechina/Leeml-Book/tree/master/docs/chapter5)
- [第4课 概率生成模型](https://github.com/datawhalechina/Leeml-Book/tree/master/docs/chapter8)
- [第5课 逻辑回归](https://github.com/datawhalechina/Leeml-Book/tree/master/docs/chapter9)


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


