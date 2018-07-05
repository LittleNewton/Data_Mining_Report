# �������
from pandas import read_csv
import pandas as pd
from sklearn import datasets
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #Ҫע�����һ��������seaborn��matplotlib��Ĭ����ͼ���ͻᱻ���ǳ�seaborn�ĸ�ʽ
%matplotlib notebook

# ��������
breast_cancer_data =pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',header=None,names = ['C_D','C_T','U_C_Si','U_C_Sh','M_A','S_E_C_S','B_N','B_C','N_N','M','Class'])

#��ʾ����ά��
print (breast_cancer_data.shape)

breast_cancer_data.info()
breast_cancer_data.head(25)  # ����ע��id 1057013 ��B_NΪ��ֵ���ã����档

print(breast_cancer_data.describe())

print(breast_cancer_data.groupby('Class').size())

mean_value = breast_cancer_data[breast_cancer_data["B_N"] != "?"]["B_N"].astype(np.int).mean() # �����쳣ֵ�е�ƽ��ֵ

mean_value = breast_cancer_data[breast_cancer_data["B_N"] != "?"]["B_N"].astype(np.int).mean() # �����쳣ֵ�е�ƽ��ֵ

breast_cancer_data["B_N"] = breast_cancer_data["B_N"].astype(np.int64)

# ����ͼ
breast_cancer_data.plot(kind='box', subplots=True, layout=(3,4), sharex=False, sharey=False)
pyplot.show()

# ֱ��ͼ
breast_cancer_data.hist()
pyplot.show()

# ɢ�����ͼ
scatter_matrix(breast_cancer_data)
pyplot.show()

# �������ݼ�
array = breast_cancer_data.values
X = array[:, 1:9] # C_DΪ��ţ���Y������ԣ����˵�
Y = array[:, 10]


validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# �㷨���
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()

num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
# �����㷨
results = []
for name in models:
    result = cross_val_score(models[name], X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(result)
    msg = '%s: %.3f (%.3f)' % (name, result.mean(), result.std())
    print(msg)
    
# ͼ����ʾ
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(models.keys())
pyplot.show()

#ʹ���������ݼ������㷨
knn = KNeighborsClassifier()
knn.fit(X=X_train, y=Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))