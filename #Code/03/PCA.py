from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
iris = datasets.load_iris()

data1=pd.DataFrame(np.concatenate((iris.data,iris.target.reshape(150,1)),axis=1),columns=np.append(iris.feature_names,'target'))
data=pd.DataFrame(np.concatenate((iris.data,np.repeat(iris.target_names,50).reshape(150,1)),axis=1), columns=np.append(iris.feature_names,'target'))
data=data.apply(pd.to_numeric,errors='ignore')

sns.pairplot(data.iloc[:,[0,1,4]],hue='target')
sns.pairplot(data.iloc[:,2:5],hue='target')

plt.scatter(data1.iloc[:,0],data1.iloc[:,1],c=data1.target)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')

from sklearn.decomposition import PCA
x_reduced = PCA(n_components=3).fit_transform(data.iloc[:,:4])
x_reduced

fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(x_reduced[:,0],x_reduced[:,1],x_reduced[:,2],c=data1.iloc[:,4])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

PC=PCA(n_components=4).fit(data.iloc[:,:4])
PC.explained_variance_ratio_