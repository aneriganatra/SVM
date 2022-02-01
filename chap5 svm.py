#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# In[2]:


iris=datasets.load_iris()
x=iris["data"][:,(2,3)]
y=(iris["target"]==2).astype(np.float64)


# In[3]:


iris


# In[4]:


svm_clf=Pipeline([("scaler",StandardScaler()),("linear_svc",LinearSVC(C=1,loss="hinge"))])
svm_clf.fit(x,y)
                  


# In[5]:


svm_clf.predict([[1,3]])


# In[6]:


from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


# In[7]:


x,y=make_moons(n_samples=100,noise=0.15)
polynomial_svm_clf=Pipeline([("poly_feature",PolynomialFeatures(degree=3)),("scaler",StandardScaler()),("linear_svc",LinearSVC(C=1,loss="hinge"))])
polynomial_svm_clf.fit(x,y)


# In[8]:


polynomial_svm_clf.predict([[1.4,1]])


# In[9]:


from sklearn.svm import SVC
poly_kernel_svm_clf=Pipeline([("scaler",StandardScaler()),("svm_clf",SVC(kernel="poly",degree=3,coef0=1,C=5))])
poly_kernel_svm_clf.fit(x,y)


# In[10]:


poly_kernel_svm_clf.predict([(5,8)])


# In[11]:


rbf_kernel_svm_clf=Pipeline([("scaler",StandardScaler()),("svm_clf",SVC(kernel="rbf",gamma=5,C=0.001))])
rbf_kernel_svm_clf.fit(x,y)


# In[12]:


rbf_kernel_svm_clf.predict([(5.5,1.5)])


# In[13]:


from sklearn.svm import LinearSVR   #better for large training data than SVR
svm_reg=LinearSVR(epsilon=1.5)
svm_reg.fit(x,y)
svm_reg.predict([(5.5,1.5)])


# In[14]:


from sklearn.svm import SVR
svm_poly_reg=SVR(kernel="poly",degree=2,C=100,epsilon=0.1)
svm_poly_reg.fit(x,y)
svm_poly_reg.predict([(5,8)])


# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(x,y)


# In[ ]:


ax=plt.gca()
xlim=ax.get_xlim()
ylim=ax.get_ylim()


# In[ ]:


xx=np.linspace(xlim[0],xlim[1],30)
yy=np.linspace(ylim[0],ylim[1],30)
YY,XX=np.meshgrid(yy,xx)
xy=np.vstack([XX.ravel(),YY.ravel()]).T
Z=clf.descision_function(xy).reshape(XX.shape)


# In[19]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[21]:


log_clf=LogisticRegression
rnd_clf=RandomForestClassifier
svm_clf=SVC()


# In[22]:


voting_clf=VotingClassifier(estimators=[('lr',log_clf),('rf',rnd_clf),('svc',svm_clf)],voting='hard')


# In[24]:


voting_clf.fit(x_train,y_train)


# In[ ]:




