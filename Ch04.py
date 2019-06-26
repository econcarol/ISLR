# ISLR Ch 4 by Carol Cui
%reset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# ----------------------------------------------------------------------------
# Q10
Weekly = pd.read_csv('C:\\Users\\Carol\\Desktop\\Weekly.csv')

# (a)
Weekly.describe()
pd.crosstab(index=Weekly["Direction"], columns="count")
Weekly.corr() # Volume increases in year.

# (b)
import statsmodels.api as sm

x01 = sm.add_constant(Weekly.iloc[:, 2:8]) 
y01 = np.where(Weekly['Direction']=='Up', 1, 0) 

glm0 = sm.Logit(y01, x01)
print(glm0.fit().summary())
# Lag2 is statistically significant.

# (c)
x = pd.DataFrame(Weekly, columns=Weekly.columns[2:8]) 
y = Weekly['Direction']

glm1      = LogisticRegression()
glm1.pred = glm1.fit(x, y).predict(x)
print(pd.DataFrame(confusion_matrix(y, glm1.pred), index=['y=0', 'y=1'], columns=['y_pred=0', 'y_pred=1']))
print('error rate: ', accuracy_score(y, glm1.pred)) # 56%

# (d)
train   = Weekly[Weekly['Year'] < 2009]
x_train = train.iloc[:,3]
x_train = x_train.reshape(len(x_train),1)
y_train = train.loc[:,'Direction']

test    = Weekly[Weekly['Year'] >= 2009] 
x_test  = test.iloc[:,3]
x_test  = x_test.reshape(len(x_test),1)
y_test  = test.loc[:,'Direction']

glm2      = LogisticRegression()
glm2.pred = glm2.fit(x_train, y_train).predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, glm2.pred), index=['y=0', 'y=1'], columns=['y_pred=0', 'y_pred=1']))
print('error rate: ', accuracy_score(y_test, glm2.pred)) # 62.5%

# (e) 
lda      = LinearDiscriminantAnalysis() 
lda.pred = lda.fit(x_train, y_train).predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, lda.pred), index=['y=0', 'y=1'], columns=['y_pred=0', 'y_pred=1']))
print('error rate: ', accuracy_score(y_test, lda.pred)) # 62.5%

# (f) 
qda      = QuadraticDiscriminantAnalysis()
qda.pred = qda.fit(x_train, y_train).predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, qda.pred), index=['y=0', 'y=1'], columns=['y_pred=0', 'y_pred=1']))
print('error rate: ', accuracy_score(y_test, qda.pred)) # 58.7%

# (g)
knn        = KNeighborsClassifier(n_neighbors=1)
knn.pred   = knn.fit(x_train, y_train).predict(x_test) 
print('error rate: ', accuracy_score(y_test, knn.pred)) # 49%

# (h): Logistic and LDA models are the best.

# (i)
# KNN
error_rate = np.array([]) 
k_value    = np.array([]) 
for i in (5, 10, 20):  
    knn        = KNeighborsClassifier(n_neighbors=i)
    knn.pred   = knn.fit(x_train, y_train).predict(x_test) 
    k_value    = np.append(k_value, i)
    error_rate = np.append(error_rate, 1-accuracy_score(y_test, knn.pred))

best_k = k_value[error_rate.argmin()]
print('KNN best when k=%i' %best_k)

# LDA
train            = Weekly[Weekly['Year'] < 2009]
x_train          = train.iloc[:,2:4]
x_train['Lag12'] = x_train.Lag1 * x_train.Lag2
y_train          = train.loc[:,'Direction']

test             = Weekly[Weekly['Year'] >= 2009] 
x_test           = test.iloc[:,2:4]
x_test['Lag12']  = x_test.Lag1 * x_test.Lag2
y_test           = test.loc[:,'Direction']

lda      = LinearDiscriminantAnalysis() 
lda.pred = lda.fit(x_train, y_train).predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, lda.pred), index=['y=0', 'y=1'], columns=['y_pred=0', 'y_pred=1']))
print('error rate: ', accuracy_score(y_test, lda.pred)) # 57.7%

# QDA
qda      = QuadraticDiscriminantAnalysis()
qda.pred = qda.fit(x_train, y_train).predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, qda.pred), index=['y=0', 'y=1'], columns=['y_pred=0', 'y_pred=1']))
print('error rate: ', accuracy_score(y_test, qda.pred)) # 46.2%

# ----------------------------------------------------------------------------
# Q11
Auto = pd.read_csv('C:\\Users\\Carol\\Desktop\\Auto.csv', na_values='?').dropna()

# (a)
Auto['mpg01'] = np.where(Auto['mpg'] > Auto['mpg'].median(), 1, 0) 

# (b)
pd.plotting.scatter_matrix(Auto.iloc[:,0:10], figsize=(10,10))
# select: displacement, horsepower, weight, acceleration

# (c)
x_name = ['displacement', 'horsepower', 'weight', 'acceleration']
x      = pd.DataFrame(Auto, columns=x_name)
y      = np.array(Auto['mpg01'])

np.random.seed(1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# (d) LDA
lda      = LinearDiscriminantAnalysis() 
lda.pred = lda.fit(x_train, y_train).predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, lda.pred), index=['y=0', 'y=1'], columns=['y_pred=0', 'y_pred=1']))
print('error rate: ', 1-accuracy_score(y_test, lda.pred)) # 7.6%

# (e) QDA
qda      = QuadraticDiscriminantAnalysis()
qda.pred = qda.fit(x_train, y_train).predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, qda.pred), index=['y=0', 'y=1'], columns=['y_pred=0', 'y_pred=1']))
print('error rate: ', 1-accuracy_score(y_test, qda.pred)) # 3.8%

# (f) Logit
glm      = LogisticRegression()
glm.pred = glm.fit(x_train, y_train).predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, glm.pred), index=['y=0', 'y=1'], columns=['y_pred=0', 'y_pred=1']))
print('error rate: ', 1-accuracy_score(y_test, glm.pred)) # 7.6%

# (g) KNN
error_rate = np.array([]) 
k_value    = np.array([]) 
for i in range(1, 110, 10):  
    knn        = KNeighborsClassifier(n_neighbors=i)
    knn.pred   = knn.fit(x_train, y_train).predict(x_test) 
    k_value    = np.append(k_value, i)
    error_rate = np.append(error_rate, 1-accuracy_score(y_test, knn.pred))

best_k = k_value[error_rate.argmin()]
print('KNN best when k=%i' %best_k)
# k = 31 is the best

# ----------------------------------------------------------------------------
# Q12
# (a)
def Power():
    print(2**3)

Power()

# (b)
def Power2(x, a):
    print(x**a)
    
Power2(3,8)

# (c)
Power2(10,3)  # 1000
Power2(8,17)  # 2.2518e+15
Power2(131,3) # 2248091

# (d)
def Power3(x, a):
    return x**a

# (e)
x = np.arange(1, 11, 1)
y = Power3(x,2)

fig = plt.figure() 
fig.add_subplot(2, 2, 1)
plt.scatter(x, y)
plt.title('log(x^2) vs x')
plt.xlabel('x')
plt.ylabel('log(x^2)')

ax = fig.add_subplot(2, 2, 2)
ax.set_xscale('log')
plt.scatter(x, y)
plt.title('log(x^2) vs x on xlog-scale')
plt.xlabel('x')
plt.ylabel('log(x^2)')

ax = fig.add_subplot(2, 2, 3)
ax.set_yscale('log')
plt.scatter(x, y)
plt.title('log(x^2) vs x on ylog-scale')
plt.xlabel('x')
plt.ylabel('log(x^2)')

ax = fig.add_subplot(2, 2, 4)
ax.set_xscale('log')
ax.set_yscale('log')
plt.scatter(x, y)
plt.title('log(x^2) vs x on xylog-scale')
plt.xlabel('x')
plt.ylabel('log(x^2)')

# (f)
def PlotPower(x, a):
    y = Power3(x, a)
    plt.scatter(x, y)
    plt.title('x^%.0f vs x' %a)
    plt.xlabel('x')
    plt.ylabel('x^%.0f' %a)

PlotPower(np.arange(1,11,1), 3)

# ----------------------------------------------------------------------------
# Q13
Boston = pd.read_csv('C:\\Users\\Carol\\Desktop\\Boston.csv')

Boston['crim01'] = np.where(Boston['crim'] > Boston['crim'].median(), 1, 0) 

Boston.corr() # indus, nox, age, dis, rad, tax
pd.plotting.scatter_matrix(Boston.iloc[:,2:17]) # nox, rm, dis, tax, black, lstat, medv
# pick: indus, nox, dis, tax, lstat

# data setup
x_name = ['indus', 'nox', 'dis', 'tax', 'lstat']
x      = pd.DataFrame(Boston, columns=x_name)
y      = np.array(Boston['crim01'])

np.random.seed(1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Logit
glm      = LogisticRegression()
glm.pred = glm.fit(x_train, y_train).predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, glm.pred), index=['y=0', 'y=1'], columns=['y_pred=0', 'y_pred=1']))
print('error rate: ', 1-accuracy_score(y_test, glm.pred)) # 21.7%

# LDA
lda      = LinearDiscriminantAnalysis() 
lda.pred = lda.fit(x_train, y_train).predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, lda.pred), index=['y=0', 'y=1'], columns=['y_pred=0', 'y_pred=1']))
print('error rate: ', 1-accuracy_score(y_test, lda.pred)) # 17.1%

# QDA
qda      = QuadraticDiscriminantAnalysis()
qda.pred = qda.fit(x_train, y_train).predict(x_test)
print(pd.DataFrame(confusion_matrix(y_test, qda.pred), index=['y=0', 'y=1'], columns=['y_pred=0', 'y_pred=1']))
print('error rate: ', 1-accuracy_score(y_test, qda.pred)) # 15.1%

# KNN
error_rate = np.array([]) 
k_value    = np.array([]) 
for i in range(1, 110, 10):  
    knn        = KNeighborsClassifier(n_neighbors=i)
    knn.pred   = knn.fit(x_train, y_train).predict(x_test) 
    k_value    = np.append(k_value, i)
    error_rate = np.append(error_rate, 1-accuracy_score(y_test, knn.pred))

best_k = k_value[error_rate.argmin()]
print('KNN best when k=%i' %best_k)
# k = 1 is the best