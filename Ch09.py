# ISLR Ch 9 by Carol Cui

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

# ----------------------------------------------------------------------------
# Q4
np.random.seed(1)
n = 100
p = 2
y = np.concatenate((np.repeat(1,30), np.repeat(1,30), np.repeat(2,40)))
x = np.random.rand(n,p)
x[1:30,:]  = x[1:30,:]  + 2
x[31:60,:] = x[31:60,:] - 2 
plt.scatter(x[:,0], x[:,1], c=y, cmap=mpl.cm.Paired)
plt.xlabel('x1')
plt.ylabel('x2')

xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.5, random_state=2)

# SVC
svc = SVC(C=10, kernel='linear')
svc.fit(xtrain, ytrain)
print(confusion_matrix(ytrain, svc.predict(xtrain))) # training error = 8/50 = 16%
plot_decision_regions(X=xtrain, y=ytrain, clf=svc, legend=2)
plt.xlabel('x1')
plt.ylabel('x2')

# SVM poly
svmp = SVC(C=10, kernel='poly', degree=4)
svmp.fit(xtrain, ytrain)
print(confusion_matrix(ytrain, svmp.predict(xtrain))) # training error = 0%
plot_decision_regions(X=xtrain, y=ytrain, clf=svmp, legend=2)
plt.xlabel('x1')
plt.ylabel('x2')

# SVM radial
svmr = SVC(C=10, kernel='rbf', gamma=1.0)
svmr.fit(xtrain, ytrain)
print(confusion_matrix(ytrain, svmr.predict(xtrain))) # training error = 0%
plot_decision_regions(X=xtrain, y=ytrain, clf=svmr, legend=2)
plt.xlabel('x1')
plt.ylabel('x2')

# SVM radial and poly outperform SVC w.r.t. training error.

# test errors
print('SVC test error: %.2f' %(1-svc.score(xtest,ytest)))
print('SVM poly test error: %.2f' %(1-svmp.score(xtest,ytest)))
print('SVM radial test error: %.2f' %(1-svmr.score(xtest,ytest)))
# SVM poly and radial perform eqaully well on test data.

# ----------------------------------------------------------------------------
# Q5
# (a)
np.random.seed(1)
n = 500
p = 2

x1 = np.random.uniform(0, 1, n) - 0.5
x2 = np.random.uniform(0, 1, n) - 0.5
y  = 1*(x1**2-x2**2 > 0)
df = pd.DataFrame({'x1':x1, 'x2':x2, 'y':y})

# (b)
fig1 = sns.lmplot(x='x1', y='x2', data=df, fit_reg=False, hue='y', legend=False)

# (c)
glm1     = LogisticRegression()
glm1_fit = glm1.fit(df.iloc[:,0:2], df['y'])

# (d)
glm1_pred = glm1_fit.predict(df.iloc[:,0:2])
fig2, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(df['x1'], df['x2'], c=glm1_pred, cmap=mpl.cm.Paired)
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_title('linear logit')

# (e)
df['x12'] = df['x1']**2
df['x22'] = df['x2']**2
glm2      = LogisticRegression()
glm2_fit  = glm2.fit(df.drop('y', axis=1), df['y'])

# (f)
glm2_pred = glm2_fit.predict(df.drop('y', axis=1))
ax2.scatter(df['x1'], df['x2'], c=glm2_pred, cmap=mpl.cm.Paired)
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_title('non-linear logit')

# (g)
svc = SVC(C=10, kernel='linear')
svc.fit(df.iloc[:,0:2], df['y'])

# (h)
svmp = SVC(C=10, kernel='poly', degree=2)
svmp.fit(df.iloc[:,0:2], df['y'])

# (i)
fig3   = plt.figure()
labels = ['SVC', 'SVM poly']
gs     = gridspec.GridSpec(1, 2)
for clf, lab, grd in zip([svc, svmp], labels, ([0,0], [0,1])):
    clf.fit(np.stack((x1, x2), axis=-1), y)
    ax   = plt.subplot(gs[grd[0], grd[1]])
    fig3 = plot_decision_regions(X=np.stack((x1, x2), axis=-1), y=y, clf=clf, legend=2)
    plt.title(lab)
plt.show()

# While SVC and linear logit are similar, SVM poly and non-linear logit are similar. 

# ----------------------------------------------------------------------------
# Q6
# (a)
np.random.seed(1)
n = 100
p = 2
y = np.concatenate((np.repeat(1,30), np.repeat(2,70)))
x = np.random.rand(n,p)
x[y==1,:] = x[y==1,:] + 0.9
df = pd.DataFrame({'x1':x[:,0], 'x2':x[:,1], 'y':y})

sns.lmplot(x='x1', y='x2', data=df, fit_reg=False, hue='y', legend=False)

# (b)
tune_param = [{'C': [0.1, 1, 10, 100, 1000]}]

tune_out = GridSearchCV(SVC(kernel='linear'), tune_param, cv=10, scoring='accuracy', return_train_score=True)
tune_out.fit(x, y)
tune_out.best_params_ 

print('mean CV error: %s' %(1-tune_out.cv_results_['mean_test_score']))
# best cost: 1
# CV error can go up as cost rises.

print('mean train error: %s' %(1-tune_out.cv_results_['mean_train_score']))
# best cost: 1000
# Training error never rises in cost. 

# (c)
np.random.seed(10)
ytest = np.concatenate((np.repeat(1,30), np.repeat(2,70)))
xtest = np.random.rand(n,p)
xtest[ytest==1,:] = xtest[ytest==1,:] + 0.9
dftest = pd.DataFrame({'x1':xtest[:,0], 'x2':xtest[:,1], 'y':ytest})
sns.lmplot(x='x1', y='x2', data=dftest, fit_reg=False, hue='y', legend=False)

test_err = np.zeros(5)
for i in range(0,5):
    svc = SVC(C=tune_param[0]['C'][i], kernel='linear')
    svc.fit(x, y)
    test_err[i] = 1-svc.score(xtest,ytest)

print('test error: %s' %test_err)
# best cost: 1
# Test error selection agrees with CV selection, not training error selection. 

# (d) The claim at the end of Section 9.6.1 is true. 

# ----------------------------------------------------------------------------
# Q7
Auto = pd.read_csv('C:\\Users\\Carol\\Desktop\\Auto.csv', na_values='?').dropna()

# (a)
Auto['mpg01'] = np.where(Auto['mpg'] > np.median(Auto['mpg']), 1, 0)
Auto          = Auto.drop(['mpg', 'name'], axis=1)
# scale continuous variables
var_no_scale  = ['cylinders', 'year', 'origin', 'mpg01']
var_to_scale  = ['displacement', 'horsepower', 'weight', 'acceleration']
scaled_var    = StandardScaler().fit_transform(Auto[var_to_scale]) 
temp1         = pd.DataFrame(scaled_var, columns=var_to_scale)
temp2         = Auto[var_no_scale].reset_index(drop=True)
df            = pd.concat([temp1, temp2], axis=1)
x             = df.iloc[:,:-1]
y             = df['mpg01']

# (b)
tune_param = [{'C': [0.01, 0.1, 1, 10]}]
tune_out   = GridSearchCV(SVC(kernel='linear'), tune_param, cv=5, scoring='accuracy', n_jobs=-1)
tune_out.fit(x, y)
best_svc   = tune_out.best_estimator_ 
print('best cost for SVC %s' %tune_out.best_params_) # best C=10

# (c)
tune_param = [{'C': [0.01, 0.1, 1, 10], 'degree': [2, 3, 4]}]
tune_out   = GridSearchCV(SVC(kernel='poly'), tune_param, cv=5, scoring='accuracy', n_jobs=-1)
tune_out.fit(x, y)
best_svmp  = tune_out.best_estimator_ 
print('best cost for SVM poly %s' %tune_out.best_params_) # best C=10, degree=2

tune_param = [{'C': [0.01, 0.1, 1, 10], 'gamma': [0.5, 1, 2]}]
tune_out   = GridSearchCV(SVC(kernel='rbf'), tune_param, cv=5, scoring='accuracy', n_jobs=-1)
tune_out.fit(x, y)
best_svmr  = tune_out.best_estimator_ 
print('best cost for SVM poly %s' %tune_out.best_params_) # best C=1, gamma=0.5

# (d)
pca      = PCA(n_components=2)
xreduced = pca.fit_transform(x)

model1 = SVC(C=10, kernel='linear')
clf1   = model1.fit(xreduced, y)

model2 = SVC(C=10, kernel='poly', degree=2)
clf2   = model2.fit(xreduced, y)

model3 = SVC(C=1, kernel='rbf', gamma=0.5)
clf3   = model3.fit(xreduced, y)

fig    = plt.figure()
fig.suptitle('decison surface using PCA transformed/projected features')
labels = ['SVC', 'SVM poly', 'SVM radial']
gs     = gridspec.GridSpec(3, 1)
for clf, lab, grd in zip([clf1, clf2, clf3], labels, ([0,0], [1,0], [2,0])):
    clf.fit(np.stack((xreduced[:,0], xreduced[:,1]), axis=-1), np.array(y))
    ax  = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=np.stack((xreduced[:,0], xreduced[:,1]), axis=-1), y=np.array(y), clf=clf, legend=2)
    plt.title(lab)
plt.show()

err_name = ['SVC', 'SVM poly', 'SVM radial']
err_rate = [1-best_svc.score(x, y), 1-best_svmp.score(x, y), 1-best_svmr.score(x, y)]
print(pd.DataFrame({'training error rate': err_rate}, index=err_name))

# SVM radial fits the training data best. 

# ----------------------------------------------------------------------------
# Q8
# (a)
OJ = pd.read_csv('C:\\Users\\Carol\\Desktop\\OJ.csv').drop('Unnamed: 0', axis=1)
OJ['Purchase'] = pd.factorize(OJ.Purchase)[0]
OJ['Store7']   = OJ.Store7.map({'No':0, 'Yes':1})

x = OJ.drop(['Purchase'], axis=1)
y = OJ['Purchase']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=800, random_state=1)

# scale continuous variables
x_no_scale    = ['StoreID', 'SpecialCH', 'SpecialMM', 'Store7', 'STORE']
x_to_scale    = xtrain[xtrain.columns.difference(x_no_scale)]
scaler        = StandardScaler().fit(x_to_scale) 
temp1         = pd.DataFrame(scaler.transform(x_to_scale), columns=xtrain.columns.difference(x_no_scale))
temp2         = xtrain[x_no_scale].reset_index(drop=True)
scaled_xtrain = pd.concat([temp1, temp2], axis=1)

x_to_scale    = xtest[xtest.columns.difference(x_no_scale)]
temp1         = pd.DataFrame(scaler.transform(x_to_scale), columns=xtest.columns.difference(x_no_scale))
temp2         = xtest[x_no_scale].reset_index(drop=True)
scaled_xtest  = pd.concat([temp1, temp2], axis=1)

# (b)
svc = SVC(C=0.01, kernel='linear')
svc.fit(scaled_xtrain, ytrain)

# (c)
svc_train_err = 1-svc.score(scaled_xtrain, ytrain)
svc_test_err  = 1-svc.score(scaled_xtest, ytest)
print('SVC training error: %.2f | SVC test error: %.2f' %(svc_train_err, svc_test_err))

# (d) & (e)
tune_param = [{'C': [0.01, 0.1, 1, 10]}]

tune_out = GridSearchCV(SVC(kernel='linear'), tune_param, cv=10, scoring='accuracy')
tune_out.fit(scaled_xtrain, ytrain)
best_svc = tune_out.best_estimator_ 

best_svc_train_err = 1-best_svc.score(scaled_xtrain, ytrain)
best_svc_test_err  = 1-best_svc.score(scaled_xtest, ytest)
print('best SVC training error: %.2f | best SVC test error: %.2f' %(best_svc_train_err, best_svc_test_err))

# (f)
svmr = SVC(C=0.01, kernel='rbf', gamma='auto')
svmr.fit(scaled_xtrain, ytrain)

svmr_train_err = 1-svmr.score(scaled_xtrain, ytrain)
svmr_test_err  = 1-svmr.score(scaled_xtest, ytest)
print('SVM radial training error: %.2f | SVM radial test error: %.2f' %(svmr_train_err, svmr_test_err))

tune_out   = GridSearchCV(SVC(kernel='rbf', gamma='auto'), tune_param, cv=10, scoring='accuracy')
tune_out.fit(scaled_xtrain, ytrain)
best_svmr  = tune_out.best_estimator_ 

best_svmr_train_err = 1-best_svmr.score(scaled_xtrain, ytrain)
best_svmr_test_err  = 1-best_svmr.score(scaled_xtest, ytest)
print('training error: %.2f | test error: %.2f' %(best_svmr_train_err, best_svmr_test_err))

# (g)
svmp = SVC(C=0.01, kernel='poly', degree=2)
svmp.fit(scaled_xtrain, ytrain)

svmp_train_err = 1-svmp.score(scaled_xtrain, ytrain)
svmp_test_err  = 1-svmp.score(scaled_xtest, ytest)
print('SVM poly training error: %.2f | SVM poly test error: %.2f' %(svmp_train_err, svmp_test_err))

tune_out   = GridSearchCV(SVC(kernel='poly', degree=2), tune_param, cv=10, scoring='accuracy')
tune_out.fit(scaled_xtrain, ytrain)
best_svmp  = tune_out.best_estimator_ 

best_svmp_train_err = 1-best_svmp.score(scaled_xtrain, ytrain)
best_svmp_test_err  = 1-best_svmp.score(scaled_xtest, ytest)
print('training error: %.2f | test error: %.2f' %(best_svmp_train_err, best_svmp_test_err))

# (h)
x_label = np.arange(6)
plt.bar(x_label, [svc_test_err, svmr_test_err, svmp_test_err, best_svc_test_err, best_svmr_test_err, best_svmp_test_err])
plt.xticks(x_label, ('linear', 'radial', 'poly', 'best linear', 'best radial', 'best poly'))
plt.ylabel('test error')
# SVC performs the best. 