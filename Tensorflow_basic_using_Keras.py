
# coding: utf-8

# # TensorFlow basics

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:

df = pd.read_csv('TensorFlow_FILES/DATA/fake_reg.csv')


# In[3]:

df.head()


# In[4]:

# very simple dataset
# we will treat it as a regression problem, feature 1, feature 2 and price to predict


# In[5]:

# supervised learning model


# In[6]:

sns.pairplot(df)


# In[7]:

plt.show()


# In[8]:

# feature 2 is veru corelated with price


# In[9]:

from sklearn.model_selection import train_test_split


# In[10]:

X = df[['feature1', 'feature2']].values # we will have to pass "Numpy arrays" instead of "Pandas arrays or series"
# adding .values to the dataframe returns a numpy array


# In[11]:

y = df['price'].values


# In[12]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)


# In[13]:

X_train.shape


# In[14]:

X_test.shape


# In[14]:

# normalize or scale your data


# In[15]:

# if we have really large values it could cause an error with weights


# In[15]:

from sklearn.preprocessing import MinMaxScaler


# In[17]:

help(MinMaxScaler)


# In[16]:

scaler = MinMaxScaler()


# In[17]:

scaler.fit(X_train)
#calculates the parameter it needs to perform the actual scaling later on
# standard deviation, min amd max


# In[18]:

X_train = scaler.transform(X_train) # this actually performs the transformation


# In[21]:

# we ran 'fit' only on train set because we want to prevent 'Data leakage' from the test set, 
# we dont want to assume that we have prior information fo the test set
# so we only try to fit our scalar tot he training set, and donot try to look into the test set


# In[19]:

X_test = scaler.transform(X_test)


# In[20]:

X_train


# In[21]:

X_train.max()


# In[22]:

X_train.min()


# In[26]:

# it has been scaled now


# In[27]:

# time to create our neural network


# In[23]:

from tensorflow.keras.models import Sequential


# In[24]:

from tensorflow.keras.layers import Dense


# In[25]:

#help(Sequential)


# In[26]:

# there is 2 ways to making a Keras based model


# In[36]:

# 1 way to do this is:
model = Sequential([Dense(4, activation='relu'), # Layer 1, 4 neurons, activation function = Relu
                   Dense(2, activation='relu'), # Layer 2, 2 neurons, activation function = Relu
                   Dense(1)]) # output layer


# In[38]:

# other way to do this is:
model = Sequential() # empty sequential model

model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))
# easier to turn off a layer in this


# In[27]:

model = Sequential() # empty sequential model

model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='rmsprop', loss='mse')


# In[28]:

model.fit(x=X_train,y=y_train,epochs=250)


# In[29]:

loss_df = pd.DataFrame(model.history.history)


# In[30]:

loss_df.plot()
plt.show()


# In[31]:

# how well this model peroforms on test data


# In[33]:

# It outputs the model's Loss
model.evaluate(X_test, y_test, verbose=0)


# In[34]:

# in our case the loss metric ins MSE
# so MSE is 25.11


# In[35]:

model.evaluate(X_train, y_train, verbose=0)


# In[36]:

test_predictions = model.predict(X_test)


# In[37]:

test_predictions


# In[38]:

test_predictions = pd.Series(test_predictions.reshape(300,))


# In[39]:

test_predictions


# In[43]:

pred_df =  pd.DataFrame(y_test, columns=['Test True Y'])


# In[44]:

pred_df


# In[45]:

pred_df = pd.concat([pred_df, test_predictions], axis=1)


# In[47]:

pred_df.columns = ['Test True Y', 'Model Predictions']


# In[48]:

pred_df


# In[54]:

sns.lmplot(x = 'Test True Y', y = 'Model Predictions', data = pred_df, scatter=True, fit_reg=False)
plt.show()


# In[55]:

# to grab different error metrics


# In[56]:

from sklearn.metrics import  mean_absolute_error, mean_squared_error


# In[57]:

mean_absolute_error(pred_df['Test True Y'], pred_df['Model Predictions'])


# In[58]:

# how do i know if it is good or bad?
# that depends on training data


# In[59]:

df.describe()


# In[60]:

# here mean price is 498 $ and our mean absolute error is 1.01 which is roughly 1%, so this error is pretty good


# In[61]:

mean_squared_error(pred_df['Test True Y'], pred_df['Model Predictions'])


# In[62]:

# this is exactly same as :
# model.evaluate(X_test, y_test, verbose=0)


# In[63]:

# RMSE
mean_squared_error(pred_df['Test True Y'], pred_df['Model Predictions'])**0.5


# In[64]:

# predicting on brand new data
# i pick this gemstone from the ground
new_gem = [[998, 1000]]


# In[65]:

# first thing is, our model is trained on 'scaled features'
# so we first need to scale this new data as per our scaler


# In[68]:

new_gem = scaler.transform(new_gem)


# In[69]:

model.predict(new_gem)


# In[70]:

# we should price it at 420 $


# In[71]:

# IF your are running a very complex model that took a lot of time to train
# yout would want to make sure you save that model
from tensorflow.keras.models import load_model


# In[72]:

model.save('my_gem_model.h5')


# In[73]:

# now I can use the load model command


# In[75]:

later_model = load_model('my_gem_model.h5')


# In[76]:

later_model.predict(new_gem)


# In[ ]:

# works as well!!!

