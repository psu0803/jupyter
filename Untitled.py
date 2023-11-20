#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
whisky = pd.read_csv("whiskies.txt")
whisky["Region"] = pd.read_csv("regions.txt")
flavors = whisky.iloc[:, 2:14]
print(flavors)


# In[6]:


corr_flavors = pd.DataFrame.corr(flavors)
print(corr_flavors)


# In[7]:


import matplotlib.pyplot as plt
plt.pcolor(corr_flavors)
plt.colorbar()
plt.savefig("whiskey1")


# In[9]:


print(flavors)
print(flavors.transpose())
corr_whisky = pd.DataFrame.corr(flavors.transpose())
print(corr_whisky)


# In[10]:


corr_whisky = pd.DataFrame.corr(flavors.transpose())
print(corr_whisky)


# In[11]:


corr_whisky = pd.DataFrame.corr(flavors.transpose())
plt.figure(figsize=(10, 10))
plt.pcolor(corr_whisky)
plt.colorbar()


# In[13]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
data, labels = make_blobs(n_samples=10000, n_features=3, centers=4, random_state=1)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels)
plt.show()


# In[ ]:




