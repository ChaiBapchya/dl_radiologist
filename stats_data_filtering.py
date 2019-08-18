import pandas as pd
train=pd.read_csv('CheXpert-v1.0-small/train.csv')
# train['AP/PA'].value_count()
print("___ AP/PA feature value counts ____")
print(train['AP/PA'].value_counts())
# train[train['AP/PA']='LL']
print("___ AP/PA == 'LL' ____")
print(train[train['AP/PA']=='LL'])
# train[train['AP/PA']=='LL'][
# 'AP/PA']
# train[train['AP/PA']=='LL']['PATH']
# train[train['AP/PA']=='LL'][['PATH']]
# train[train['AP/PA']=='LL']
# train[train['AP/PA']=='LL'][0]
# train[train['AP/PA']=='LL']['Path']
pd.set_option('display.max_columns',100)
# train[train['AP/PA']=='LL']['Path']
pd.set_option('display.max_colwidth',-1)
# train[train['AP/PA']=='LL']['Path']
# train[train['AP/PA']=='LL']
# train[train['AP/PA']=='LL'].write_csv('output.csv')
# train[train['AP/PA']=='LL'].to_csv('output.csv')
# train['AP/PA'].value_counts()
# print(train[train['AP/PA']=='RL'])
# train['AP/PA'].value_counts()
# train.describe()
# train = train.filter(train['Frontal/Lateral']=='Frontal')
# train[train['AP/PA']=='RL']
# train['AP/PA'].value_counts()
# train.head()
# train=pd.read_csv('train.csv')
# train['AP/PA'].value_counts()
# train['Frontal/Lateral'].value_counts()
# train = train[train['Frontal/Lateral']=='Frontal']
# train.count()
# train['AP/PA'].value_counts()
print("------- SEX value count -----------")
print(train['Sex'].value_counts())
# train.describe()
# train[train['Age']==0]
# train[train['No Finding']==1]
# train[train['No Finding']==1].count()
