
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:

df = pd.read_csv('TensorFlow_FILES/DATA/cancer_classification.csv')


# In[4]:

df.info()# to check for nulls


# In[7]:

df.describe().transpose()


# In[9]:

sns.countplot(x='benign_0__mal_1', data=df)
plt.show()


# In[14]:

df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')
plt.show()


# In[16]:

sns.heatmap(df.corr())
plt.show()


# In[17]:

X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values


# In[18]:

from sklearn.model_selection import train_test_split


# In[20]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.27, random_state = 101)


# In[21]:

from sklearn.preprocessing import MinMaxScaler


# In[22]:

scaler = MinMaxScaler()


# In[24]:

X_train = scaler.fit_transform(X_train)


# In[25]:

X_test = scaler.transform(X_test)


# In[26]:

from tensorflow.keras.models import Sequential


# In[27]:

from tensorflow.keras.layers import Dense, Dropout


# In[29]:

X_train.shape


# In[33]:

model = Sequential()

model.add(Dense(30, activation = 'relu'))

model.add(Dense(15, activation = 'relu'))
# BINARY CLASSIFICATION
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam')


# In[35]:

model.fit(x=X_train, y=y_train, epochs=600, validation_data = (X_test,y_test))


# In[36]:

losses = pd.DataFrame(model.history.history)


# In[40]:

losses.plot()
plt.show()


# In[41]:

# this is a perfect example of overfitting
# Too many epochs
# We will use Early Stopping to correct this


# In[42]:

model = Sequential()

model.add(Dense(30, activation = 'relu'))

model.add(Dense(15, activation = 'relu'))
# BINARY CLASSIFICATION
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam')


# In[43]:

from tensorflow.keras.callbacks import EarlyStopping


# In[44]:

help(EarlyStopping)


# In[45]:

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
# patience : we will wait for 25 epochs


# In[46]:

model.fit(x=X_train, y=y_train, epochs=600, validation_data = (X_test,y_test), callbacks=[early_stop])


# In[49]:

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
plt.show()


# In[50]:

# green line flattenning out is OK behaviur, we wanto avoid its increase


# In[51]:

# ADD DROP-OUT LAYER


# In[52]:

# this will essentially turn off a percentage of Neurons automatically


# In[53]:

from tensorflow.keras.layers import Dropout


# In[54]:

model = Sequential()

model.add(Dense(30, activation = 'relu'))
model.add(Dropout(0.5)) # half of the neurn sin this layer of 30 will be turned off randomly

model.add(Dense(15, activation = 'relu'))
model.add(Dropout(0.5))

# BINARY CLASSIFICATION
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam')


# In[55]:

model.fit(x=X_train, y=y_train, epochs=600, validation_data = (X_test,y_test), callbacks=[early_stop])


# In[56]:

model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
plt.show()


# In[57]:

# much imporoveed


# In[59]:

predictions = model.predict_classes(X_test)


# In[60]:

from sklearn.metrics import classification_report, confusion_matrix


# In[61]:

print(classification_report(y_test, predictions))


# In[62]:

# very good performance


# In[63]:

print(confusion_matrix(y_test, predictions))


# In[ ]:



