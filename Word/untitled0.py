# -*- coding: utf-8 -*-
"""
Created on Wed May 4 21:09:09 2018

@author: angelshare
"""

import xlrd
import matplotlib.pyplot as plt
from numpy import *
import sklearn
from sklearn.decomposition import PCA
from random import randrange
import tensorflow as tf



file=r"C:\Users\angelshare\Documents\python\primaldata.xls"
data=xlrd.open_workbook(file)
sheet0 = data.sheet_by_index(0)
nrows=sheet0.nrows
ncols = sheet0.ncols

def featureNormalize(X):
    '''归一化数据标准差'''
    n = X.shape[1]
    mu = zeros((1,n));
    sigma = zeros((1,n))

    mu = mean(X,axis=0)
    sigma = std(X,axis=0)
    for i in range(n):
        X[:,i] = (X[:,i]-mu[i])/sigma[i]
    return (X)

datamatrix=zeros((270,ncols-1))
datamatrix2=zeros((270,ncols-1))

for x in range(ncols-1):
    cols =sheet0.col_slice(x,4,nrows)
    cols =list(map(str,cols))
    for y in range(len(cols)):
        k=cols[y]
        cols[y]=k[-3:]
    cols=list(map(float,cols))
    minVals=min(cols)
    maxVals=max(cols)
    cols1=matrix(cols)
    ranges=maxVals-minVals
    b=cols1-minVals
    normcols=b/ranges# 数据进行归一化处理
    datamatrix[:,x]=normcols
    datamatrix2[:,x]=cols1
    
datamatrix2=featureNormalize(datamatrix2)

label=sheet0.col_slice(13,4,nrows)
label=list(map(str,label))
for x in range(270):
    if label[x]== "text:'Ⅰ类'":
        label[x]=0
    else:
        label[x]=1
label=array(label)

#PCA
pca = PCA(n_components='mle')
pca.fit(datamatrix)
print(pca.explained_variance_ratio_)
plt.figure()
plt.plot(pca.explained_variance_ratio_, 'k', linewidth=2)
plt.xlabel('n_components', fontsize=16)
plt.ylabel('explained_variance_ratio_', fontsize=16)
plt.show()    

#标准差归一化方法
pca2 = PCA(n_components='mle')
pca2.fit(datamatrix2)
print(pca2.explained_variance_ratio_)
plt.figure()
plt.plot(pca2.explained_variance_ratio_, 'k', linewidth=2)
plt.xlabel('n_components', fontsize=16)
plt.ylabel('explained_variance_ratio_', fontsize=16)
plt.show()   

pca=PCA(n_components=2)
pca.fit(datamatrix)
newdata=pca.transform(datamatrix)
print('维度',newdata.shape)
print(pca.components_) # 投影方向向量
print('\n\n')

#relief 交叉验证
def distanceNorm(Norm,D_value):
	if Norm == '1':
		counter = absolute(D_value);
		counter = sum(counter);
	elif Norm == '2':
		counter = power(D_value,2);
		counter = sum(counter);
		counter = sqrt(counter);
	elif Norm == 'Infinity':
		counter = absolute(D_value);
		counter = max(counter);
	else:
		raise Exception('We will program this later......');
	return counter;

def fit(features,labels,iter_ratio):
	(n_samples,n_features) = shape(features);
	distance = zeros((n_samples,n_samples));
	weight = zeros(n_features);

	if iter_ratio >= 0.5:
		# compute distance
		for index_i in range(n_samples):
			for index_j in range(index_i+1,n_samples):
				D_value = features[index_i] - features[index_j];
				distance[index_i,index_j] = distanceNorm('2',D_value);
		distance += distance.T;
	else:
		pass;
	# start iteration
	for iter_num in range(int(iter_ratio*n_samples)):
		nearHit = list();
		nearMiss = list();
		distance_sort = list();
		# random extract a sample
		index_i = randrange(0,n_samples,1);
		self_features = features[index_i];
		# search for nearHit and nearMiss
		if iter_ratio >= 0.5:
			distance[index_i,index_i] = max(distance[index_i]);		# filter self-distance 
			for index in range(n_samples):
				distance_sort.append([distance[index_i,index],index,labels[index]]);
		else:
			# compute distance respectively
			distance = zeros(n_samples);
			for index_j in range(n_samples):
				D_value = features[index_i] - features[index_j];
				distance[index_j] = distanceNorm('2',D_value);
			distance[index_i] = max(distance);		# filter self-distance 
			for index in range(n_samples):
				distance_sort.append([distance[index],index,labels[index]]);
		distance_sort.sort(key = lambda x:x[0]);
		for index in range(n_samples):
			if nearHit == [] and distance_sort[index][2] == labels[index_i]:
				# nearHit = distance_sort[index][1];
				nearHit = features[distance_sort[index][1]];
			elif nearMiss == [] and distance_sort[index][2] != labels[index_i]:
				# nearMiss = distance_sort[index][1]
				nearMiss = features[distance_sort[index][1]];
			elif nearHit != [] and nearMiss != []:
				break;
			else:
				continue;

		# update weight

		weight = weight - power(self_features - nearHit,2) + power(self_features - nearMiss,2);
	print (weight/(iter_ratio*n_samples));
	return (weight/(iter_ratio*n_samples));

weight=fit(datamatrix2,label,300)


#svm分类

sess = tf.Session()

x_vals = newdata
y_vals = label

train_indices = random.choice(len(x_vals),round(len(x_vals)*0.8),replace=False)
test_indices = array(list(set(range(len(x_vals))) - set(train_indices)))

x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

batch_size = 100

# 初始化feedin
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 创建变量
A = tf.Variable(tf.random_normal(shape=[2, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# 定义线性模型
model_output = tf.subtract(tf.matmul(x_data, A), b)

# Declare vector L2 'norm' function squared
l2_norm = tf.reduce_sum(tf.square(A))

# Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
alpha = tf.constant([0.01])
classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

# Training loop
loss_vec = []
train_accuracy = []
test_accuracy = []
for i in range(20000):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    
[[a1], [a2]] = sess.run(A)
[[b]] = sess.run(b)
slope = -a2/a1
y_intercept = b/a1
best_fit = []

x1_vals = [d[1] for d in x_vals]

for i in x1_vals:
    best_fit.append(slope*i+y_intercept)



setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == 1]
setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == 1]
not_setosa_x = [d[1] for i, d in enumerate(x_vals) if y_vals[i] == -1]
not_setosa_y = [d[0] for i, d in enumerate(x_vals) if y_vals[i] == -1]

plt.plot(setosa_x, setosa_y, 'o', label='I 型')
plt.plot(not_setosa_x, not_setosa_y, 'x', label='II型')
plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=3)
plt.show()