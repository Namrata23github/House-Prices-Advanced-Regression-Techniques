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


# In[3667]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[3668]:


houses=pd.concat([train,test], sort=False)


# In[3669]:


houses.select_dtypes(include='object').head()


# In[3670]:


houses.select_dtypes(include=['float','int']).head()


# In[3671]:


houses.select_dtypes(include='object').isnull().sum()[houses.select_dtypes(include='object').isnull().sum()>0]


# In[3672]:


houses.select_dtypes(include=['int','float']).isnull().sum()[houses.select_dtypes(include=['int','float']).isnull().sum()>0]


# In[3673]:


for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
           'PoolQC','Fence','MiscFeature'):
    train[col]=train[col].fillna('None')
    test[col]=test[col].fillna('None')


# In[3674]:


for col in ('MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):
    train[col]=train[col].fillna(train[col].mode()[0])
    test[col]=test[col].fillna(train[col].mode()[0])


# In[3675]:


for col in ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea'):
    train[col]=train[col].fillna(0)
    test[col]=test[col].fillna(0)


# In[3676]:


train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())
test['LotFrontage']=test['LotFrontage'].fillna(train['LotFrontage'].mean())


# In[3677]:


print(train.isnull().sum().sum())
print(train.isnull().sum().sum())


# In[3678]:


# plt.figure(figsize=[40,20])
# sns.heatmap(train.corr(), annot=True)
print(train.shape)
print(test.shape)


# In[3679]:


train.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)
test.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)


# In[3680]:


train = train[train['GrLivArea']<4000]


# In[3681]:


len_train=train.shape[0]
print(train.shape)


# In[3682]:


houses=pd.concat([train,test], sort=False)


# In[3683]:


houses['MSSubClass']=houses['MSSubClass'].astype(str)


# In[3684]:


skew=houses.select_dtypes(include=['int','float']).apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skew_df=pd.DataFrame({'Skew':skew})
skewed_df=skew_df[(skew_df['Skew']>0.5)|(skew_df['Skew']<-0.5)]


# In[3685]:


skewed_df.index


# In[3686]:


train=houses[:len_train]
test=houses[len_train:]


# In[3687]:


# lam=0.1
# for col in ('MiscVal', 'PoolArea', 'LotArea', 'LowQualFinSF', '3SsnPorch',
#        'KitchenAbvGr', 'BsmtFinSF2', 'EnclosedPorch', 'ScreenPorch',
#        'BsmtHalfBath', 'MasVnrArea', 'OpenPorchSF', 'WoodDeckSF',
#        'LotFrontage', 'GrLivArea', 'BsmtFinSF1', 'BsmtUnfSF', 'Fireplaces',
#        'HalfBath', 'TotalBsmtSF', 'BsmtFullBath', 'OverallCond', 'YearBuilt',
#        'GarageYrBlt'):
#     train[col]=boxcox1p(train[col],lam)
#     test[col]=boxcox1p(test[col],lam)


# In[ ]:





# In[3688]:


train['SalePrice']


# In[ ]:





# In[3689]:


houses


# In[ ]:





# In[ ]:





# In[3690]:


lam=0.1
for col in ('MiscVal', 'PoolArea', 'LotArea', 'LowQualFinSF', '3SsnPorch',
       'KitchenAbvGr', 'BsmtFinSF2', 'EnclosedPorch', 'ScreenPorch',
       'BsmtHalfBath', 'MasVnrArea', 'OpenPorchSF', 'WoodDeckSF',
       'LotFrontage', 'GrLivArea', 'BsmtFinSF1', 'BsmtUnfSF', 'Fireplaces',
       'HalfBath', 'TotalBsmtSF', 'BsmtFullBath', 'OverallCond', 'YearBuilt',
       'GarageYrBlt'):
    train[col]=boxcox1p(train[col],lam)
    test[col]=boxcox1p(test[col],lam)


# In[3691]:


houses=pd.concat([train,test], sort=False)
houses=pd.get_dummies(houses)


# In[ ]:





# In[3692]:


train=houses[:len_train]
test=houses[len_train:]


# In[3693]:


houses.dtypes


# In[3694]:


train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)


# In[3695]:


y= np.log1p(train["SalePrice"])
x=train.drop('SalePrice', axis=1)
test=test.drop('SalePrice', axis=1)


# In[3696]:


test


# In[3697]:


from sklearn.preprocessing import StandardScaler
x.dtypes


# In[3698]:



x = StandardScaler().fit_transform(x)
test =  StandardScaler().fit_transform(test)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size = 0.2)


# In[3699]:


print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)

X_train


# In[3700]:


# # Feature Scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_val = sc.transform(X_val)


# In[3701]:


X_train.shape[1]


# In[3702]:


# from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
# from sklearn.model_selection import GridSearchCV
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import KFold
# from sklearn.model_selection import cross_val_score

# def build_regressor():
#     regressor = Sequential()
#     regressor.add(Dense(units = 5000, kernel_initializer = 'normal', activation = 'sigmoid', input_dim = 314))
#     regressor.add(Dropout(rate = 0.1))
#     regressor.add(Dense(units = 400, kernel_initializer = 'normal', activation = 'sigmoid'))
#     regressor.add(Dropout(rate = 0.1))
#     regressor.add(Dense(units = 300, kernel_initializer = 'normal', activation = 'sigmoid'))
#     regressor.add(Dropout(rate = 0.1))
#     regressor.add(Dense(units = 100, kernel_initializer = 'normal', activation = 'relu'))
#     regressor.add(Dropout(rate = 0.1))
#     regressor.add(Dense(units = 10, kernel_initializer = 'normal', activation = 'relu'))
#     regressor.add(Dropout(rate = 0.1))
#     regressor.add(Dense(units = 1,kernel_initializer = 'normal'))
#     regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
#     return regressor


# In[3703]:


from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers


# #Model1
# model = Sequential()
# #model.add(Dense(256, activation="relu", input_dim = X_train.shape[1]))
# model.add(Dense(1, input_dim = X_train.shape[1], W_regularizer=l1(0.001)))

# model.compile(loss = "mse", optimizer = "adam")or


# In[3704]:


from keras.layers.normalization import BatchNormalization

#Model2
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


# In[3705]:


# p = {'lr': (0.5, 5, 10),
#      'first_neuron':[4, 8, 16, 32, 64],
#      'hidden_layers':[0, 1, 2],
#      'batch_size': (2, 30, 10),
#      'epochs': [150],
#      'dropout': (0, 0.5, 5),
#      'weight_regulizer':[None],
#      'emb_output_dims': [None],
#      'shape':['brick','long_funnel'],
#      'optimizer': [Adam, Nadam, RMSprop],
#      'losses': [logcosh, binary_crossentropy],
#      'activation':[relu, elu],
#      'last_activation': [sigmoid]}


# In[3706]:


model = build_regressor()
model.summary()
early_stopping = tf.keras.callbacks.EarlyStopping()

# regressor = KerasRegressor(build_fn = build_regressor)
# parameters = {'batch_size': [100, 300],
#               'epochs': [100, 500],
#               'optimizer': ['adam', 'rmsprop']}
# grid_search = GridSearchCV(estimator = regressor,
#                            param_grid = parameters,
#                            scoring = 'accuracy',
#                            cv = 10)
# grid_search = grid_search.fit(X_train, y_train)
# best_parameters = grid_search.best_params_
# best_accuracy = grid_search.best_score_
# grid_search = grid_search.fit(X_train, y_train)
hist = model.fit(X_train, y_train, epochs=500,batch_size = 80,callbacks=[early_stopping], validation_data = (X_val, y_val))
scores = np.sqrt(model.evaluate(X_val,y_val,verbose=2))
scores


# In[3707]:


# early_stopping = tf.keras.callbacks.EarlyStopping()
# model = build_regressor()


# In[3708]:


# estimators = []
# estimators.append(('standardize', StandardScaler()))
# estimators.append(('mlp', KerasRegressor(build_fn=build_regressor, epochs=100, batch_size=5, verbose=2)))
# pipeline = Pipeline(estimators)
# kfold = KFold(n_splits=10)
# results = cross_val_score(pipeline, x, y, cv=kfold)
# print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# In[3722]:


# pd.Series(model.predict(X_val)[:,0]).hist()
PRED = model.predict(X_val)
fig, ax = plt.subplots(figsize=(30,50))
ax.plot( range(0, PRED.shape[0]), PRED[:,0])
ax.plot( range(0, y_val.shape[0]), y_val.values)


# In[3710]:


# model=Lasso(alpha =0.001, random_state=1)


# In[3711]:


# model.fit(x,y)


# In[3723]:


np.log(0.0372)


# In[3713]:


predictions = model.predict(test)


# In[3714]:


predictions


# In[3715]:


pred = list(predictions)
predictions


# In[3716]:


final_value  = np.expm1(predictions)


# In[3717]:


type(final_value)
final_value =np.ravel(final_value)


# In[3718]:


testId = pd.read_csv('test.csv')
testId.columns


# In[3719]:


final_df = pd.DataFrame({'Id': testId['Id'], 'SalePrice': final_value})


# In[3720]:


final_df.to_csv('salePricePrediction.csv', header=True, index=False)



# In[ ]:




