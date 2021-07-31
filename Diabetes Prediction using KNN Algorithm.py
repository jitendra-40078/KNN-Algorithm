#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# In[6]:


df = pd.read_csv('diabetes.csv')
df.head()


# In[11]:


non_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for coloumn in non_zero:
    df[coloumn] = df[coloumn].replace(0, np.NaN)
    mean = int(df[coloumn].mean(skipna = True))
    df[coloumn] = df[coloumn].replace(np.NaN, mean)
    print(df[coloumn])


# In[17]:


import seaborn as sns
p = sns.pairplot(df, hue= 'Outcome')


# In[18]:


X= df.iloc[:, 0:8]
y= df.iloc[:, 8]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =42, stratify = y)


# In[19]:


sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


# In[20]:


import math
math.sqrt(len(y_test))


# In[25]:


clf = KNeighborsClassifier(n_neighbors = 10, p =2, metric = 'euclidean')
clf.fit(X_train, y_train)


# In[26]:


y_pred = clf.predict(X_test)
y_pred


# In[27]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[28]:


print(f1_score(y_test, y_pred))


# In[29]:


print(accuracy_score(y_test, y_pred))


# In[33]:


import matplotlib.pyplot as plt
plt.figure(figsize=(5, 7))
ax = sns.distplot(df['Outcome'], hist=False, color="r", label="Actual Value")
sns.distplot(y_pred, hist=False, color="b", label="Predicted Values", ax=ax)
plt.title('Actual vs Precited value for outcome')
plt.show()
plt.close()


# In[ ]:




