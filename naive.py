#!/usr/bin/env python
# coding: utf-8

# 
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# 

# In[21]:


import pandas as pd 
df = pd.read_csv('iris.csv')
df


# In[13]:


df.shape


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, train_size=0.8)


# In[24]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
pred


# In[29]:


import seaborn as sns

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,pred)
sns.heatmap(cm)


# In[32]:


import  matplotlib.pyplot as plt
plt.xlabel('predict label')
plt.ylabel('actual label')
plt.title('confusion matrix')
plt.show()


# In[13]:


print(cm)


# In[34]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[35]:


print('accuracy score : ',accuracy_score(y_test,y_pred))


# In[18]:


print('classification_report : ',classification_report(y_test,y_pred))


# In[ ]:




