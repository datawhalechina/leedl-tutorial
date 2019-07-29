import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
lr=LogisticRegression()

train_data=pd.read_csv('./data/train.csv')
test_data=pd.read_csv('./data/test.csv')


train_data['workclass'][train_data.workclass=='?']='Private'
test_data['workclass'][train_data['workclass']=='?']='Private'
train_data['occupation'][train_data['occupation']=='?']='other'
test_data['occupation'][test_data['occupation']=='?']='other'

cols=['workclass', 'education','marital-status', 'occupation', 'relationship', 'race', 'sex']
def p_data(data,col):
    tmp=pd.get_dummies(data[col])
    data=pd.concat([data,tmp],axis=1)
    data=data.drop(col,axis=1)
    return data
for col in cols:
    train_data=p_data(train_data,col)
    test_data=p_data(test_data,col)

data=pd.concat([train_data,test_data],axis=0)
mp=data['native-country'].value_counts()/data.shape[0]
data['native-country']=data['native-country'].map(mp)

data[['age','fnlwgt' ,'education-num' ,'capital-gain' ,'capital-loss' ,'hours-per-week']]=min_max_scaler.fit_transform(data[['age','fnlwgt' ,'education-num' ,'capital-gain' ,'capital-loss' ,'hours-per-week']])

data['lable']=data['lable'].