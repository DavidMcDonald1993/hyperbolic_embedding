
# coding: utf-8

# In[8]:

get_ipython().magic(u'matplotlib inline')
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# In[3]:

X, y = load_iris(return_X_y=True)


# In[4]:

X.shape


# In[12]:

Y =(X - np.mean(X, axis=0)) / np.std(X, axis=0)


# In[5]:

np.cov(X)


# In[15]:

plt.scatter(Y[:,0], Y[:,1], c=y)


# In[14]:

np.std(Y, axis=0)


# In[25]:

np.cov(Y)


# In[27]:

Y.dot(Y.T)


# In[29]:

from sklearn.decomposition import PCA


# In[55]:

pca = PCA(n_components=4)
Z = pca.fit_transform(Y)


# In[60]:

plt.scatter(Zn[:,0], Zn[:,1], c=y)


# In[56]:

U = pca.components_.T


# In[57]:

U


# In[58]:

U.dot(U.T)


# In[59]:

U.T.dot(U)


# In[61]:

pca.explained_variance_


# In[62]:

Zn = Z / np.sqrt(pca.explained_variance_)


# In[63]:

np.cov(Zn.T)


# In[71]:

Zn = Zn.dot(U)


# In[72]:

np.cov(Zn.T)


# In[ ]:



