#-*- coding:utf-8 -*-
# @File    : PM2.5Prediction.py
# @Date    : 2019-05-19
# @Author  : 追风者
# @Software: PyCharm
# @Python Version: python 3.6

'''
利用线性回归Linear Regression模型预测 PM2.5

特征工程中的特征选择与数据可视化的直观分析
通过选择的特征进一步建立回归模型
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''数据读取与预处理'''
# DataFrame类型
train_data = pd.read_csv("./Dataset/train.csv")
train_data.drop(['Date', 'stations', 'observation'], axis=1, inplace=True)

ItemNum=18
#训练样本features集合
X_Train=[]
#训练样本目标PM2.5集合
Y_Train=[]

for i in range(int(len(train_data)/ItemNum)):
    observation_data = train_data[i*ItemNum:(i+1)*ItemNum] #一天的观测数据
    for j in range(15):
        x = observation_data.iloc[:, j:j + 9]
        y = int(observation_data.iloc[9,j+9])
        # 将样本分别存入X_Train、Y_Train中
        X_Train.append(x)
        Y_Train.append(y)
# print(X_Train)
# print(Y_Train)

'''绘制散点图'''
x_AMB=[]
x_CH4=[]
x_CO=[]
x_NMHC=[]

# x_NO=[]
# x_NO2=[]
# x_NOX=[]
# x_O3=[]

# x_PM10=[]
# x_PM2Dot5=[]
# x_RAINFALL=[]
# x_RH=[]

# x_SO2=[]
# x_THC=[]
# x_WD_HR=[]
# x_WIND_DIREC=[]

# x_WIND_SPEED=[]
# x_WS_HR=[]
#
y=[]
#
# for i in range(len(Y_Train)):
#     y.append(Y_Train[i])
#     x=X_Train[i]
#     # print(type(x.iloc[0,0]))
#     # 求各测项的平均值
#     x_WIND_SPEED_sum = 0
#     x_WS_HR_sum = 0
#     for j in range(9):
#         x_WIND_SPEED_sum = x_WIND_SPEED_sum + float(x.iloc[0, j])
#         x_WS_HR_sum = x_WS_HR_sum + float(x.iloc[1, j])
#     x_WIND_SPEED.append(x_WIND_SPEED_sum / 9)
#     x_WS_HR.append(x_WS_HR_sum / 9)
# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# plt.title('WIND_SPEED')
# plt.scatter(x_WIND_SPEED, y)
# plt.subplot(1, 2, 2)
# plt.title('WS_HR')
# plt.scatter(x_WS_HR, y)
# plt.show()
#     x_SO2_sum = 0
#     x_THC_sum = 0
#     x_WD_HR_sum = 0
#     x_WIND_DIREC_sum = 0
#     for j in range(9):
#         x_SO2_sum = x_SO2_sum + float(x.iloc[0, j])
#         x_THC_sum = x_THC_sum + float(x.iloc[1, j])
#         x_WD_HR_sum = x_WD_HR_sum + float(x.iloc[2, j])
#         x_WIND_DIREC_sum = x_WIND_DIREC_sum + float(x.iloc[3, j])
#     x_SO2.append(x_SO2_sum / 9)
#     x_THC.append(x_THC_sum / 9)
#     x_WD_HR.append(x_WD_HR_sum / 9)
#     x_WIND_DIREC.append(x_WIND_DIREC_sum / 9)
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 2, 1)
# plt.title('SO2')
# plt.scatter(x_SO2, y)
# plt.subplot(2, 2, 2)
# plt.title('THC')
# plt.scatter(x_THC, y)
# plt.subplot(2, 2, 3)
# plt.title('WD_HR')
# plt.scatter(x_WD_HR, y)
# plt.subplot(2, 2, 4)
# plt.title('WIND_DIREC')
# plt.scatter(x_WIND_DIREC, y)
# plt.show()
#     x_PM10_sum = 0
#     x_PM2Dot5_sum = 0
#     x_RAINFALL_sum = 0
#     x_RH_sum = 0
#     for j in range(9):
#         x_PM10_sum = x_PM10_sum + float(x.iloc[0, j])
#         x_PM2Dot5_sum = x_PM2Dot5_sum + float(x.iloc[1, j])
#         x_RAINFALL_sum = x_RAINFALL_sum + float(x.iloc[2, j])
#         x_RH_sum = x_RH_sum + float(x.iloc[3, j])
#     x_PM10.append(x_PM10_sum / 9)
#     x_PM2Dot5.append(x_PM2Dot5_sum / 9)
#     x_RAINFALL.append(x_RAINFALL_sum / 9)
#     x_RH.append(x_RH_sum / 9)
# plt.figure(figsize=(10, 6))
# plt.subplot(2, 2, 1)
# plt.title('PM10')
# plt.scatter(x_PM10, y)
# plt.subplot(2, 2, 2)
# plt.title('PM2.5')
# plt.scatter(x_PM2Dot5, y)
# plt.subplot(2, 2, 3)
# plt.title('RAINFALL')
# plt.scatter(x_RAINFALL, y)
# plt.subplot(2, 2, 4)
# plt.title('RH')
# plt.scatter(x_RH, y)
# plt.show()
    x_AMB_sum=0
    x_CH4_sum=0
    x_CO_sum=0
    x_NMHC_sum=0
    for j in range(9):
        x_AMB_sum = x_AMB_sum + float(x.iloc[0,j])
        x_CH4_sum = x_CH4_sum + float(x.iloc[1, j])
        x_CO_sum = x_CO_sum + float(x.iloc[2, j])
        x_NMHC_sum = x_NMHC_sum + float(x.iloc[3, j])
    x_AMB.append(x_AMB_sum / 9)
    x_CH4.append(x_CH4_sum / 9)
    x_CO.append(x_CO_sum / 9)
    x_NMHC.append(x_NMHC_sum / 9)
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.title('AMB')
plt.scatter(x_AMB, y)
plt.subplot(2, 2, 2)
plt.title('CH4')
plt.scatter(x_CH4, y)
plt.subplot(2, 2, 3)
plt.title('CO')
plt.scatter(x_CO, y)
plt.subplot(2, 2, 4)
plt.title('NMHC')
plt.scatter(x_NMHC, y)
plt.show()

'''小批量梯度下降'''
dict={0:8,1:8,2:8,3:8,4:8,5:8,6:8,7:8,8:8,9:9,10:9,11:9,12:9,13:9,14:9,15:9,16:9,17:9,18:12,19:12,20:12,21:12,22:12,23:12,24:12,25:12,26:12}
iteration_count = 10000   #迭代次数
learning_rate = 0.000001  #学习速率
b=0.0001    #初始化偏移项
parameters=[0.001]*27     #初始化27个参数
loss_history=[]
for i in range(iteration_count):
    loss=0
    b_grad=0
    w_grad=[0]*27
    examples=list(np.random.randint(0, len(X_Train)-1) for index in range(100))
    for j in range(100):
        index=examples.pop()
        day = X_Train[index]
        partsum = b+float(parameters[0]*day.iloc[8,0])+float(parameters[1]*day.iloc[8,1])+\
                  float(parameters[2]*day.iloc[8,2])+float(parameters[3]*day.iloc[8,3])+ \
                  float(parameters[4]*day.iloc[8,4])+float(parameters[5]*day.iloc[8,5])+ \
                  float(parameters[6]*day.iloc[8,6])+float(parameters[7]*day.iloc[8,7])+ \
                  float(parameters[8]*day.iloc[8,8])+float(parameters[9]*day.iloc[9,0])+ \
                  float(parameters[10]*day.iloc[9,1])+float(parameters[11]*day.iloc[9,2])+ \
                  float(parameters[12]*day.iloc[9,3])+float(parameters[13]*day.iloc[9,4])+ \
                  float(parameters[14]*day.iloc[9,5])+float(parameters[15]*day.iloc[9,6])+ \
                  float(parameters[16]*day.iloc[9,7])+float(parameters[17]*day.iloc[9,8])+ \
                  float(parameters[18]*day.iloc[12,0])+float(parameters[19]*day.iloc[12,1])+ \
                  float(parameters[20]*day.iloc[12,2])+float(parameters[21]*day.iloc[12,3])+ \
                  float(parameters[22]*day.iloc[12,4])+float(parameters[23]*day.iloc[12,5])+ \
                  float(parameters[24]*day.iloc[12,6])+float(parameters[25]*day.iloc[12,7])+ \
                  float(parameters[26]*day.iloc[12,8]) - Y_Train[index]
        loss = loss + partsum * partsum
        b_grad = b_grad + partsum
        for k in range(27):
            w_grad[k]=w_grad[k]+ partsum * day.iloc[dict[k],k % 9]
    loss_history.append(loss/2)
    #更新参数
    b = b - learning_rate * b_grad
    for t in range(27):
        parameters[t] = parameters[t] - learning_rate * w_grad[t]


# '''评价模型'''
# data1 = pd.read_csv('./Dataset/test.csv')
# del data1['id']
# del data1['item']
# X_Test=[]
# ItemNum=18
# for i in range(int(len(data1)/ItemNum)):
#     day = data1[i*ItemNum:(i+1)*ItemNum] #一天的观测数据
#     X_Test.append(day)
# Y_Test=[]
# data2 = pd.read_csv('./Dataset/answer.csv')
# for i in range(len(data2)):
#     Y_Test.append(data2.iloc[i,1])
# b=0.00371301266193
# parameters=[-0.0024696993501677625, 0.0042664323568029619, -0.0086174899917209787, -0.017547874680980298, -0.01836289806786489, -0.0046459546176775678, -0.031425910733080147, 0.018037490234208024, 0.17448898242705385, 0.037982590870111861, 0.025666115101346722, 0.02295437149703404, 0.014272058968395849, 0.011573452230087483, 0.010984971346586308, -0.0061003639742210781, 0.19310213021199321, 0.45973205224805752, -0.0034995637680653086, 0.00094072189075279807, 0.00069329550591916357, 0.002966257320079194, 0.0050690506276038138, 0.007559004246038563, 0.013296350700555241, 0.027251049329127801, 0.039423988570899793]
# Y_predict=[]
# for i in range(len(X_Test)):
#     day=X_Test[i]
#     p=b+parameters[0]*day.iloc[8,0]+parameters[1]*day.iloc[8,1]+parameters[2]*day.iloc[8,2]+parameters[3]*day.iloc[8,3]+parameters[4]*day.iloc[8,4]+parameters[5]*day.iloc[8,5]+parameters[6]*day.iloc[8,6]+parameters[7]*day.iloc[8,7]+parameters[8]*day.iloc[8,8]+parameters[9]*day.iloc[9,0]+parameters[10]*day.iloc[9,1]+parameters[11]*day.iloc[9,2]+parameters[12]*day.iloc[9,3]+parameters[13]*day.iloc[9,4]+parameters[14]*day.iloc[9,5]+parameters[15]*day.iloc[9,6]+parameters[16]*day.iloc[9,7]+parameters[17]*day.iloc[9,8]+parameters[18]*day.iloc[12,0]+parameters[19]*day.iloc[12,1]+parameters[20]*day.iloc[12,2]+parameters[21]*day.iloc[12,3]+parameters[22]*day.iloc[12,4]+parameters[23]*day.iloc[12,5]+parameters[24]*day.iloc[12,6]+parameters[25]*day.iloc[12,7]+parameters[26]*day.iloc[12,8]
#     Y_predict.append(p)
# def dev_degree(y_true,y_predict):    #评价函数
#     sum=0
#     for i in range(len(y_predict)):
#         sum=sum+(y_true[i]-y_predict[i])*(y_true[i]-y_predict[i])
#     return sum/len(y_predict)
# print(dev_degree(Y_Test,Y_predict))