#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# Load the training, additional, confidence, and test data
train_data = pd.read_csv('training.csv')
test_data = pd.read_csv('testing.csv')
additional_data = pd.read_csv('additional_training.csv')
confidence = pd.read_csv('annotation_confidence.csv')


# In[3]:


# Look at their shapes
print('Train data: ',train_data.shape)
print('Test data: ',test_data.shape)
print('Additional data: ',additional_data.shape)
print('Confidence data: ',confidence.shape)


# In[4]:


# Visualise some of the train_data
train_data.tail(2)


# In[5]:


# Visualise some of the additional_data
additional_data.tail(2)


# In[6]:


# Make sure the additional and training data are of the same type so we can add them together
print(train_data.dtypes)
print(additional_data.dtypes)


# In[649]:


# Since they're the same types, append them together.
full_train = train_data.append(additional_data)
print(full_train.shape)
full_train


# In[650]:


# Fill the NaN values
full_train_ = full_train.fillna(full_train.mean())


# In[651]:


np.where(np.isnan(full_train_))


# In[652]:


full_train_['confidence'] = confidence['confidence']


# In[653]:


predictions = full_train_.prediction
confidences = full_train_.confidence
train = full_train_.drop('prediction',axis=1)
train = train.drop('confidence',axis=1)
train.tail(1)


# In[654]:


from sklearn.model_selection import train_test_split
x_train,x_val = train_test_split(full_train_,train_size=0.7)


# In[655]:


np.where(np.isnan(x_train))


# In[656]:


np.where(np.isnan(x_val))


# In[657]:


train_pred = x_train.prediction
val_pred = x_val.prediction
train_conf = x_train.confidence
val_conf = x_val.confidence

x_train = x_train.drop('prediction',axis=1)
x_train = x_train.drop('confidence',axis=1)
x_val = x_val.drop('prediction',axis=1)
x_val = x_val.drop('confidence',axis=1)


# In[721]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_val_scaled = scaler.transform(x_val)
test_scaled = scaler.transform(test_data)


# In[755]:


from sklearn.decomposition import PCA
pca2 = PCA().fit(x_train_scaled)
plt.plot(np.cumsum(pca2.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')


# In[722]:


pca = PCA(1500)
pca.fit(x_train_scaled)
pcaTrain = pca.transform(x_train_scaled)
pcaVal = pca.transform(x_val_scaled)
pcaTest = pca.transform(test_scaled)


# In[723]:


train_2 = scaler.fit_transform(train)
train_2 = pca.transform(train_2)


# In[724]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg2 = LogisticRegression(C=1)
logreg2.fit(pcaTrain, train_pred ,sample_weight=train_conf)


# In[725]:


valPred = logreg2.predict(pcaVal)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg2.score(pcaVal, val_pred)))


# In[726]:


np.where(valPred==0)


# In[727]:


np.where(val_pred==0)


# In[715]:


test_y = logreg2.predict(pcaTest)


# In[716]:


np.where(test_y==0)


# In[ ]:





# In[736]:


log_df = pd.DataFrame(test_y)
log_df.columns = ['prediction']
predds=pd.DataFrame({'ID': test_data.ID, 'prediction': log_df['prediction']})
predds.to_csv('predictions.csv', index=False)
predds


# In[748]:


from sklearn import model_selection 
from sklearn.ensemble import BaggingClassifier
from sklearn import svm

seed = 8
kfold = model_selection.KFold(n_splits = 10, random_state = seed)

# initialize the base classifier 
base_cls = LogisticRegression(C=1)
  
# no. of base classifier 
num_trees = 10

# bagging classifier 
model = BaggingClassifier(base_estimator = base_cls, n_estimators = num_trees, random_state = seed) 


# In[749]:


model.fit(train_2, predictions ,sample_weight=confidences)


# In[750]:


y = model.predict(pcaVal)


# In[751]:


print('Accuracy of bagging classifier classifier on test set: {:.2f}'.format(model.score(pcaVal, val_pred)))


# In[752]:


y_test = model.predict(pcaTest)


# In[753]:


np.where(y_test==0)


# In[754]:


log_df = pd.DataFrame(y_test)
log_df.columns = ['prediction']
predds=pd.DataFrame({'ID': test_data.ID, 'prediction': log_df['prediction']})
predds.to_csv('ajnfjnefk done.csv', index=False)
predds



from sklearn import model_selection 
from sklearn.ensemble import BaggingClassifier
from sklearn import svm

seed = 8
kfold = model_selection.KFold(n_splits = 10, random_state = seed)

# initialize the base classifier 
base_cls = LogisticRegression(C=1,penalty='elasticnet',solver='saga',l1_ratio=0.5)
  
# no. of base classifier 
num_trees = list(range(1,10))
accuracy = []
# bagging classifier 
for n in num_trees:
    model = BaggingClassifier(base_estimator = base_cls, n_estimators = n, random_state = seed) 
    model.fit(train_2, predictions ,sample_weight=confidences)
    acc = model.score(pcaVal, val_pred)
    accuracy.append(acc)


# In[778]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(num_trees, accuracy)
plt.xlabel("Number of estimators")
plt.ylabel("Accuracy")
plt.show()





