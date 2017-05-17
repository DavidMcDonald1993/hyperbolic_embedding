
# coding: utf-8

# In[1]:

import numpy as np

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA


# In[9]:

X = np.array([[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]])
Y = np.array([[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]])
cca = CCA(n_components=1)
cca.fit(X, Y)


# In[14]:

X_c, Y_c = cca.transform(X, Y)


# In[15]:

X_c.shape


# In[18]:

X_c


# In[17]:

Y_c


# In[19]:

from scipy.stats import pearsonr


# In[26]:

r, p = pearsonr(X_c, Y_c)


# In[28]:

r
p


# In[24]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[25]:

plt.scatter(X_c, Y_c)


# In[29]:

from sklearn.datasets import load_digits


# In[30]:

X, y = load_digits(return_X_y=True)


# In[31]:

X.shape
y.shape


# In[49]:

digit_cca = CCA(n_components=2, max_iter=1000, tol=1e-8)


# In[50]:

digit_cca.fit(X, y)


# In[51]:

X_c, y_c = digit_cca.transform(X, y)


# In[53]:

y_c.shape


# In[57]:

X


# In[54]:

plt.scatter(X_c[:,0], X_c[:,1], c=y)


# In[41]:

X_c.shape
y_c.shape


# In[47]:

pearsonr(X_c.flatten(), y_c)


# In[46]:

X_c.flatten()


# In[44]:

y_c


# In[58]:

get_ipython().system(u'head reactome_edgelist.txt')


# In[ ]:



