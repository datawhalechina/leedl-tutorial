# Winner or Loser
本作业主要是使用逻辑回归来判断一个人的年薪是否大于50k
## 数据集和任务描述
- 任务：二分类问题，判断一个人年薪是否超过50k

- 数据集：ADULT

	由Barry Becker从1994年人口普查数据库中提取，按照((AGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0))条件过滤后，保持数据的干净。
	
	参考：https://archive.ics.uci.edu/ml/datasets/Adult

## 特征属性描述
数据一共包含13个特征，一个年薪是否超过50K的label
train.csv 、test.csv :
age, workclass, fnlwgt, education, education num, marital-status, occupation
relationship, race, sex, capital-gain, capital-loss, hours-per-week,
native-country, make over 50K a year or not
![12-1](./res/chapter12-1.png)
		
## 抽取后的特征
- 离散数据进行one-hot编码，如work_class,education...
- 连续特征保持不变，如age,capital_gain...
- X_train,X_test 每个样本包含106维特征，一个样本作为一行
- Y_train:label=0 表示年薪低于等于50k,label=1 表示年薪高于50K
![12-2](./res/chapter12-2.png)


# 参考代码
https://github.com/datawhalechina/Leeml-Book/tree/master/docs/Homework/HW-2




