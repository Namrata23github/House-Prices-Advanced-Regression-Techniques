#!/usr/bin/env python
# coding: utf-8

# In[3666]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso, LassoCV

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
houses=pd.concat([train,test], sort=False)
houses.select_dtypes(include='object').head()
houses.select_dtypes(include=['float','int']).head()
houses.select_dtypes(include='object').isnull().sum()[houses.select_dtypes(include='object').isnull().sum()>0]
houses.select_dtypes(include=['int','float']).isnull().sum()[houses.select_dtypes(include=['int','float']).isnull().sum()>0]


for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
           'PoolQC','Fence','MiscFeature'):
    train[col]=train[col].fillna('None')
    test[col]=test[col].fillna('None')

for col in ('MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):
    train[col]=train[col].fillna(train[col].mode()[0])
    test[col]=test[col].fillna(train[col].mode()[0])
            
for col in ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea'):
    train[col]=train[col].fillna(0)
    test[col]=test[col].fillna(0)

train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())
test['LotFrontage']=test['LotFrontage'].fillna(train['LotFrontage'].mean())

print(train.isnull().sum().sum())
print(train.isnull().sum().sum())

train.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)
test.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)

train = train[train['GrLivArea']<4000]
len_train=train.shape[0]
houses=pd.concat([train,test], sort=False)

houses['MSSubClass']=houses['MSSubClass'].astype(str)

skew=houses.select_dtypes(include=['int','float']).apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skew_df=pd.DataFrame({'Skew':skew})
skewed_df=skew_df[(skew_df['Skew']>0.5)|(skew_df['Skew']<-0.5)]
skewed_df.index

train=houses[:len_train]
test=houses[len_train:]

lam=0.1
for col in ('MiscVal', 'PoolArea', 'LotArea', 'LowQualFinSF', '3SsnPorch',
       'KitchenAbvGr', 'BsmtFinSF2', 'EnclosedPorch', 'ScreenPorch',
       'BsmtHalfBath', 'MasVnrArea', 'OpenPorchSF', 'WoodDeckSF',
       'LotFrontage', 'GrLivArea', 'BsmtFinSF1', 'BsmtUnfSF', 'Fireplaces',
       'HalfBath', 'TotalBsmtSF', 'BsmtFullBath', 'OverallCond', 'YearBuilt',
       'GarageYrBlt'):
    train[col]=boxcox1p(train[col],lam)
    test[col]=boxcox1p(test[col],lam)
            
houses=pd.concat([train,test], sort=False)
houses=pd.get_dummies(houses)
train=houses[:len_train]
test=houses[len_train:]
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

y= np.log1p(train["SalePrice"])
x=train.drop('SalePrice', axis=1)
test=test.drop('SalePrice', axis=1)

from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)
test =  StandardScaler().fit_transform(test)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size = 0.2)

from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers


from keras.layers.normalization import BatchNormalization

def build_regressor():
    model = Sequential()
    BatchNormalization()
    model.add(Dense(1500,input_dim=X_train.shape[1],activation='relu',kernel_initializer='normal', W_regularizer=regularizers.l2(0.01)))
    BatchNormalization()
    Dropout(0.3)
    model.add(Dense(1500,input_dim=X_train.shape[1],activation='relu',kernel_initializer='normal', W_regularizer=regularizers.l1(0.001)))
    BatchNormalization()
    Dropout(0.3)
    model.add(Dense(1300,input_dim=X_train.shape[1],activation='relu',kernel_initializer='normal', W_regularizer=regularizers.l1(0.001)))
    BatchNormalization()
    Dropout(0.3)
    model.add(Dense(1300,input_dim=X_train.shape[1],activation='relu',kernel_initializer='normal',W_regularizer=regularizers.l1(0.001)))
    BatchNormalization()
    Dropout(0.2)
    model.add(Dense(50))
    BatchNormalization()
    model.add(Dense(1))
    model.compile(optimizer="adam",loss='mean_squared_error',metrics=['mean_squared_error'])
    return model


model = build_regressor()
model.summary()
early_stopping = tf.keras.callbacks.EarlyStopping()
hist = model.fit(X_train, y_train, epochs=500,batch_size = 80,callbacks=[early_stopping], validation_data = (X_val, y_val))
scores = np.sqrt(model.evaluate(X_val,y_val,verbose=2))
scores

PRED = model.predict(X_val)
fig, ax = plt.subplots(figsize=(30,50))
ax.plot( range(0, PRED.shape[0]), PRED[:,0])
ax.plot( range(0, y_val.shape[0]), y_val.values)

predictions = model.predict(test)
pred = list(predictions)
final_value  = np.expm1(predictions)
final_value =np.ravel(final_value)
testId = pd.read_csv('test.csv')
testId.columns
final_df = pd.DataFrame({'Id': testId['Id'], 'SalePrice': final_value})
final_df.to_csv('salePricePrediction.csv', header=True, index=False)


