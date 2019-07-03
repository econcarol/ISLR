# ISLR Ch 6 by Carol Cui

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures, scale
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------------------
# Q8
# (a)
np.random.seed(1234)
n = 100
X = np.random.randn(n)
e = np.random.randn(n)

# (b)
b = np.array([0.5, 1.2, 0.8, 3.6])
Y = b[0] + b[1]*X + b[2]*X**2 + b[3]*X**3 + e
plt.scatter(X,Y)

# (c)
p      = 10
poly   = PolynomialFeatures(degree=p)
x_poly = poly.fit_transform(X.reshape(-1, 1))
xnames = []
for i in range(1,11): xnames.append('x'+str(i))
x_poly = pd.DataFrame(x_poly[:,1:], columns=xnames)

# best subset selection
def bestsub(dfx, dfy, i):
    xcombo = []
    xnum   = []
    coef   = []
    adjr2  = []
    bic    = []
    mse    = []
    
    for combo in itertools.combinations(dfx.columns,i):
        tempx = sm.add_constant(dfx[list(combo)])
        lm    = sm.OLS(dfy,tempx).fit()
        xcombo.append(list(combo))
        xnum.append(len(combo))
        coef.append(list(lm.params))
        adjr2.append(lm.rsquared_adj)
        bic.append(lm.bic)
        mse.append(lm.mse_resid)
        
    return(pd.DataFrame({'xcombo':xcombo, 'xnum':xnum, 'coef':coef, 
                         'adj_R2':adjr2, 'BIC':bic, 'MSE':mse}))

results = pd.DataFrame()  
for i in range(1,p+1):
    results = results.append(bestsub(x_poly, Y, i))

# adjusted R2
max_adjr2  = results.groupby(['xnum']).adj_R2.max()
best_adjr2 = max_adjr2.idxmax(axis=1)
plt.plot(max_adjr2.index, max_adjr2)
plt.scatter(best_adjr2, max_adjr2[best_adjr2], s=100, c='r')
plt.xlabel('# of features')
plt.ylabel('adjusted R-squared')

best_adjr2_xcombo = results[results['adj_R2']==max_adjr2[best_adjr2]].xcombo
print(best_adjr2_xcombo)
best_adjr2_coef   = results[results['adj_R2']==max_adjr2[best_adjr2]].coef
print(best_adjr2_coef.values.reshape(-1,1))

# BIC
min_bic  = results.groupby(['xnum']).BIC.min()
best_bic = min_bic.idxmin(axis=1)
plt.plot(min_bic.index, min_bic)
plt.scatter(best_bic, min_bic[best_bic], s=100, c='r')
plt.xlabel('# of features')
plt.ylabel('BIC')

best_bic_xcombo = results[results['BIC']==min_bic[best_bic]].xcombo
print(best_bic_xcombo)
best_bic_coef   = results[results['BIC']==min_bic[best_bic]].coef
print(best_bic_coef.values.reshape(-1,1))

# Cp
mse_all            = float(results[results['xnum']==p].MSE)
results['MSE_all'] = mse_all
results['Cp']      = results['xnum'] + (results['MSE']-results['MSE_all'])*(n-results['xnum'])/results['MSE_all']

min_cp  = results.groupby(['xnum']).Cp.min()
best_cp = min_cp.idxmin(axis=1)
plt.plot(min_cp.index, min_cp)
plt.scatter(best_cp, min_cp[best_cp], s=100, c='r')
plt.xlabel('# of features')
plt.ylabel('Cp')

best_cp_xcombo = results[results['Cp']==min_cp[best_cp]].xcombo
print(best_cp_xcombo)
best_cp_coef   = results[results['Cp']==min_cp[best_cp]].coef
print(best_cp_coef.values.reshape(-1,1))

# Adj R2, BIC and Cp all agree on the true model. 

# (d)
# forward stepwise selection
oldxs  = []
xcombo = []
coef   = []
adjr2  = []
bic    = []
mse    = []
ssr    = []

run     = 0
remainx = x_poly

for i in range(1, p+1):
       
    best_ssr = np.inf
    
    for combo in itertools.combinations(remainx.columns,1):
        tempx = sm.add_constant(x_poly[oldxs + list(combo)])
        lm    = sm.OLS(Y,tempx).fit()
        run  += 1 # record # of models ran (should be p(p+1)/2)
        
        if lm.ssr < best_ssr:
            addx       = list(combo)[0]
            best_coef  = list(lm.params)
            best_adjr2 = lm.rsquared_adj
            best_bic   = lm.bic
            best_mse   = lm.mse_resid
            best_ssr   = lm.ssr
            
    # a list includes all the x's that have been chosen
    # this updates the list, so later we must make a copy for recording
    oldxs.append(addx)
    # remove selected x's from the dataset
    remainx = x_poly.drop(labels=list(oldxs), axis=1)
        
    # record info
    xcombo.append(oldxs.copy())
    coef.append(best_coef)    
    adjr2.append(best_adjr2)
    bic.append(best_bic)
    mse.append(best_mse)
    ssr.append(best_ssr)

print(run)

results = pd.DataFrame({'xnum':range(1,p+1), 'xcombo':xcombo, 'coef':coef,
                        'adj_R2':adjr2, 'BIC':bic, 'MSE':mse, 'RSS':ssr})

# adjusted R2
max_adjr2 = results.adj_R2.max()
plt.plot(range(1,p+1), results['adj_R2'])
plt.scatter(results.adj_R2.argmax()+1, max_adjr2, s=100, c='r')
plt.xlabel('# of features')
plt.ylabel('adjusted R-squared')

best_adjr2_xcombo = results[results['adj_R2']==max_adjr2].xcombo
print(best_adjr2_xcombo)
best_adjr2_coef   = results[results['adj_R2']==max_adjr2].coef
print(best_adjr2_coef.values.reshape(-1,1))

# BIC
min_bic = results.BIC.min()
plt.plot(range(1,p+1), results['BIC'])
plt.scatter(results.BIC.argmin()+1, min_bic, s=100, c='r')
plt.xlabel('# of features')
plt.ylabel('BIC')

best_bic_xcombo = results[results['BIC']==min_bic].xcombo
print(best_bic_xcombo)
best_bic_coef   = results[results['BIC']==min_bic].coef
print(best_bic_coef.values.reshape(-1,1))

# Cp
results['MSE_all'] = mse_all
results['Cp']      = results['xnum'] + (results['MSE']-results['MSE_all'])*(n-results['xnum'])/results['MSE_all']

min_cp = results.Cp.min()
plt.plot(range(1,p+1), results['Cp'])
plt.scatter(results.Cp.argmin()+1, min_cp, s=100, c='r')
plt.xlabel('# of features')
plt.ylabel('Cp')

best_cp_xcombo = results[results['Cp']==min_cp].xcombo
print(best_cp_xcombo)
best_cp_coef   = results[results['Cp']==min_cp].coef
print(best_cp_coef.values.reshape(-1,1))

# All pick 4-feature model including (x1,x2,x3,x4).

# backward stepwise selection
xcombo = []
coef   = []
adjr2  = []
bic    = []
mse    = []
ssr    = []

run     = 0
remainx = x_poly

for i in range(p, 0, -1):
       
    best_ssr = np.inf
    
    for combo in itertools.combinations(remainx.columns,i):
        tempx = sm.add_constant(x_poly[list(combo)])
        lm    = sm.OLS(Y,tempx).fit()
        run  += 1 
        
        if lm.ssr < best_ssr:
            keepx      = list(combo)
            best_coef  = list(lm.params)
            best_adjr2 = lm.rsquared_adj
            best_bic   = lm.bic
            best_mse   = lm.mse_resid
            best_ssr   = lm.ssr
            
    # keep selected x's in the dataset
    remainx = x_poly[keepx]
        
    # record info
    xcombo.append(keepx)
    coef.append(best_coef)    
    adjr2.append(best_adjr2)
    bic.append(best_bic)
    mse.append(best_mse)
    ssr.append(best_ssr)

print(run)

results = pd.DataFrame({'xnum':range(p,0,-1), 'xcombo':xcombo, 'coef':coef,
                        'adj_R2':adjr2, 'BIC':bic, 'MSE':mse, 'RSS':ssr})

# adjusted R2
max_adjr2 = results.adj_R2.max()
plt.plot(range(p,0,-1), results['adj_R2'])
plt.scatter(results.loc[results.adj_R2.argmax(),'xnum'], max_adjr2, s=100, c='r')
plt.xlabel('# of features')
plt.ylabel('adjusted R-squared')

best_adjr2_xcombo = results[results['adj_R2']==max_adjr2].xcombo
print(best_adjr2_xcombo)
best_adjr2_coef   = results[results['adj_R2']==max_adjr2].coef
print(best_adjr2_coef.values.reshape(-1,1))

# BIC
min_bic = results.BIC.min()
plt.plot(range(p,0,-1), results['BIC'])
plt.scatter(results.loc[results.BIC.argmin(),'xnum'], min_bic, s=100, c='r')
plt.xlabel('# of features')
plt.ylabel('BIC')

best_bic_xcombo = results[results['BIC']==min_bic].xcombo
print(best_bic_xcombo)
best_bic_coef   = results[results['BIC']==min_bic].coef
print(best_bic_coef.values.reshape(-1,1))

# Cp
results['MSE_all'] = mse_all
results['Cp']      = results['xnum'] + (results['MSE']-results['MSE_all'])*(n-results['xnum'])/results['MSE_all']

min_cp = results.Cp.min()
plt.plot(range(p,0,-1), results['Cp'])
plt.scatter(results.loc[results.Cp.argmin(),'xnum'], min_cp, s=100, c='r')
plt.xlabel('# of features')
plt.ylabel('Cp')

best_cp_xcombo = results[results['Cp']==min_cp].xcombo
print(best_cp_xcombo)
best_cp_coef   = results[results['Cp']==min_cp].coef
print(best_cp_coef.values.reshape(-1,1))

# Adj R2 and Cp pick 6-feature model (x1,x3,x4,x6,x8,x10), but BIC picks 4-feature model (x1,x3,x4,x6).

# (e)
# 10-fold CV
lassocv = LassoCV(cv=10, normalize=True) 
lassocv.fit(x_poly, Y)
print('best alpha: %.4f' %lassocv.alpha_) #0.0139
print('intercept: %.2f' %lassocv.intercept_)
print('coefficient:') 
print(lassocv.coef_)

# plot MSE as a fn of alpha
m_log_alphas = -np.log10(lassocv.alphas_)
plt.figure()
plt.plot(m_log_alphas, lassocv.mse_path_, ':')
plt.plot(m_log_alphas, lassocv.mse_path_.mean(axis=-1), 'k', label='average across the folds', linewidth=2)
plt.axvline(-np.log10(lassocv.alpha_), linestyle='--', color='k', label='alpha: CV estimate')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('mean square error')
plt.axis('tight')

# LASSO selected model that is close to the true model.

# (f)
# generate data
b = np.array([0.5, 4.6])
Y = b[0] + b[1]*X**7 + e
plt.scatter(X,Y)

# best subset selection
results = pd.DataFrame()  
for i in range(1,p+1):
    results = results.append(bestsub(x_poly, Y, i))

# adjusted R2
max_adjr2  = results.groupby(['xnum']).adj_R2.max()
best_adjr2 = max_adjr2.idxmax(axis=1)
plt.plot(max_adjr2.index, max_adjr2)
plt.scatter(best_adjr2, max_adjr2[best_adjr2], s=100, c='r')
plt.xlabel('# of features')
plt.ylabel('adjusted R-squared')

best_adjr2_xcombo = results[results['adj_R2']==max_adjr2[best_adjr2]].xcombo
print(best_adjr2_xcombo)
best_adjr2_coef   = results[results['adj_R2']==max_adjr2[best_adjr2]].coef
print(best_adjr2_coef.values.reshape(-1,1))

# BIC
min_bic  = results.groupby(['xnum']).BIC.min()
best_bic = min_bic.idxmin(axis=1)
plt.plot(min_bic.index, min_bic)
plt.scatter(best_bic, min_bic[best_bic], s=100, c='r')
plt.xlabel('# of features')
plt.ylabel('BIC')

best_bic_xcombo = results[results['BIC']==min_bic[best_bic]].xcombo
print(best_bic_xcombo)
best_bic_coef   = results[results['BIC']==min_bic[best_bic]].coef
print(best_bic_coef.values.reshape(-1,1))

# Cp
mse_all            = float(results[results['xnum']==p].MSE)
results['MSE_all'] = mse_all
results['Cp']      = results['xnum'] + (results['MSE']-results['MSE_all'])*(n-results['xnum'])/results['MSE_all']

min_cp  = results.groupby(['xnum']).Cp.min()
best_cp = min_cp.idxmin(axis=1)
plt.plot(min_cp.index, min_cp)
plt.scatter(best_cp, min_cp[best_cp], s=100, c='r')
plt.xlabel('# of features')
plt.ylabel('Cp')

best_cp_xcombo = results[results['Cp']==min_cp[best_cp]].xcombo
print(best_cp_xcombo)
best_cp_coef   = results[results['Cp']==min_cp[best_cp]].coef
print(best_cp_coef.values.reshape(-1,1))

# BIC and Cp select the true model, and Adj R2 is close. 

# LASSO based on LOOCV
lassocv = LassoCV(cv=n, normalize=True) 
lassocv.fit(x_poly, Y)
print('best alpha: %.4f' %lassocv.alpha_) #0.0139
print('intercept: %.2f' %lassocv.intercept_)
print('coefficient:') 
print(lassocv.coef_)

# plot MSE as a fn of alpha
m_log_alphas = -np.log10(lassocv.alphas_)
plt.figure()
plt.plot(m_log_alphas, lassocv.mse_path_, ':')
plt.plot(m_log_alphas, lassocv.mse_path_.mean(axis=-1), 'k', label='average across the folds', linewidth=2)
plt.axvline(-np.log10(lassocv.alpha_), linestyle='--', color='k', label='alpha: CV estimate')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('mean square error')
plt.axis('tight')

# LASSO selected model greatly differs from the true model. 

# ----------------------------------------------------------------------------
# Q9
College     = pd.read_csv('C:\\Users\\Carol\\Desktop\\college.csv')
school_name = College.iloc[:,0] 
College     = College.rename(index = lambda x: school_name[x]) 
College.drop(College.columns[[0]], axis=1, inplace=True)

College['Private01'] = np.where(College.Private=='Yes', 1, 0)
College.drop(['Private'], axis = 1, inplace=True)

# (a)
X = College.drop(['Apps'], axis=1).astype('float64')
Y = College['Apps']
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.5, random_state=1)

# (b)
lm      = LinearRegression().fit(xtrain, ytrain)
lm_pred = lm.predict(xtest)
lm_err  = mean_squared_error(ytest, lm_pred) #1425056

# (c)
alphas     = 10**np.linspace(10,-2,100)*0.5
ridgecv    = RidgeCV(alphas=alphas, normalize = True).fit(xtrain, ytrain) #LOOCV
ridge_pred = ridgecv.predict(xtest)
ridge_err  = mean_squared_error(ytest, ridge_pred) #1534640
ridge_coef = pd.Series(ridgecv.coef_, index = xtrain.columns)
print(ridge_coef)

# (d)
lassocv    = LassoCV(cv=len(ytrain), normalize=True).fit(xtrain, ytrain) #LOOCV
lasso_pred = lassocv.predict(xtest)
lasso_err  = mean_squared_error(ytest, lasso_pred) #1412065
lasso_coef = pd.Series(lassocv.coef_, index = xtrain.columns)
print(lasso_coef)
print('# of non-zero coef estimates: %i' %lasso_coef[lasso_coef != 0].count()) #16

# (e)
# scale data
pca = PCA()
reduced_xtrain = pca.fit_transform(scale(xtrain))
# amount of variance explained by adding each principal component
np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

# LOOCV with shuffle
ntrain = len(reduced_xtrain)
loocv  = KFold(n_splits=ntrain, shuffle=True, random_state=1)

# PCR analysis
# pick M based on MSE
pcr    = LinearRegression()
mse    = []
# compute MSE with only intercept (no principal components)
score = -1*cross_val_score(pcr, np.ones((ntrain,1)), ytrain.ravel(), cv=loocv, scoring='neg_mean_squared_error').mean()    
mse.append(score)
# compute MSE with the 17 principle components, adding one at a time
for i in np.arange(1,18):
    score = -1*cross_val_score(pcr, reduced_xtrain[:,:i], ytrain.ravel(), cv=loocv, scoring='neg_mean_squared_error').mean()
    mse.append(score)
# plot MSE against # of components
plt.plot(mse, '-o')
plt.xlabel('# of principal components')
plt.ylabel('MSE')
# select M
M = mse.index(min(mse)) # index is 0 for intercept-only MSE
print('M = %i' %M) #16

# predict using M and test data
reduced_xtest = pca.transform(scale(xtest))[:,:M+1]

pcr.fit(reduced_xtrain[:,:M+1], ytrain)
pcr_pred = pcr.predict(reduced_xtest)
pcr_err  = mean_squared_error(ytest, pcr_pred) #2158674

# (f)
# scale data
scaled_xtrain = scale(xtrain)

# PLS analysis
# pick M based on MSE
mse = []
for i in np.arange(1,18):
    pls   = PLSRegression(n_components=i)
    score = -1*cross_val_score(pls, scaled_xtrain, ytrain, cv=loocv, scoring='neg_mean_squared_error').mean()
    mse.append(score)
# plot MSE against # of components
plt.plot(mse, '-o')
plt.xlabel('# of principal components')
plt.ylabel('MSE')
# select M
M = mse.index(min(mse))+1 # index is 0 for intercept-only MSE (aka 1 component)
print('M = %i' %M) #9

pls      = PLSRegression(n_components=M).fit(scaled_xtrain, ytrain)
pls_pred = pls.predict(scale(xtest))
pls_err  = mean_squared_error(ytest, pls_pred) #2159278

# (g)
err = pd.DataFrame({'errors': [lm_err, ridge_err, lasso_err, pcr_err, pls_err]})
err.rename(index={0:'lm', 1:'ridge', 2:'lasso', 3:'pcr', 4:'pls'}, inplace=True) 
print(err)

x_label = np.arange(5)
plt.bar(x_label, err.errors)
plt.xticks(x_label, ('lm', 'ridge', 'lasso', 'pcr', 'pls'))
plt.ylabel('test MSE')

# There is a difference: LASSO performs the best and PLS the worst.

# ----------------------------------------------------------------------------
# 10
# (a)
np.random.seed(1234)
n = 1000 # # of obs
p = 20   # # of features

e = np.random.randn(n) # error term

# coefficients w/ some non-zeros
b        = np.zeros(p) 
i        = np.random.choice(range(0,20), size=5, replace=False)
b[i]     = [12, 3.5, 1.4, 0.6, 0.05]
true_mod = pd.DataFrame({'b': b[i]}, index=i+1).rename_axis('xi', axis=1).sort_index()

# predictors and response
X = np.random.randn(n,p)
Y = np.matmul(X,b) + e

# put X and Y into data frame
Xnames = []
for i in range(1,p+1): Xnames.append('x'+str(i))
dfx = pd.DataFrame(X, columns=Xnames)
dfy = pd.DataFrame({'y': Y})

# (b)
np.random.seed(1)
x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size=0.9)

# (c)
# Use forward stepwise selection to save run time
oldxs  = []
xcombo = []
coef   = []
mse    = []
ssr    = []
err    = []

remainx = x_train

for i in range(1, p+1):
       
    best_ssr = np.inf
    
    for combo in itertools.combinations(remainx.columns,1):
        tempx = sm.add_constant(x_train[oldxs + list(combo)])
        lm    = sm.OLS(y_train,tempx).fit()
        
        if lm.ssr < best_ssr:
            addx       = list(combo)[0]
            best_coef  = list(lm.params)
            best_mse   = ((lm.predict(tempx)-y_train.y)**2).mean()
            best_ssr   = lm.ssr
            temp_xtest = sm.add_constant(x_test[tempx.columns[1:]])
            test_mse   = ((lm.predict(temp_xtest)-y_test.y)**2).mean()
            
    # a list includes all the x's that have been chosen
    # this updates the list, so later we must make a copy for recording
    oldxs.append(addx)
    # remove selected x's from the dataset
    remainx = x_train.drop(labels=list(oldxs), axis=1)
        
    # record info
    xcombo.append(oldxs.copy())
    coef.append(best_coef)    
    mse.append(best_mse)
    ssr.append(best_ssr)
    err.append(test_mse)

results = pd.DataFrame({'xnum':range(1,p+1), 'xcombo':xcombo, 'coef':coef,
                        'MSE':mse, 'RSS':ssr})

# training MSE
min_mse = results.MSE.min()
plt.plot(range(1,p+1), results['MSE'])
plt.scatter(results.MSE.idxmin(axis=1)+1, min_mse, s=100, c='r')
plt.xlabel('# of features')
plt.ylabel('training MSE')

# (d)
plt.plot(range(1,p+1), err)
plt.scatter(err.index(min(err))+1, min(err), s=100, c='r')
plt.xticks(np.arange(1, 21, step=1))
plt.xlabel('# of features')
plt.ylabel('testing MSE')

# (e)
print('testing MSE select model size %i' %(err.index(min(err))+1)) #4

# (f)
best_xcombo = results[results['xnum']==(err.index(min(err))+1)].xcombo
print(best_xcombo)
best_coef   = results[results['xnum']==(err.index(min(err))+1)].coef
print(best_coef.values.reshape(-1,1))
print(true_mod)
# Estimates are close to the true model.

# (g)
values = []
dfb    = pd.DataFrame({'b':b}, index=Xnames)
for r in range(0,p):
    dfc = pd.DataFrame({'coef':results.coef[r][1:]}, index=results.xcombo[r])
    df  = pd.merge(dfb, dfc, how="outer", left_index=True, right_index=True).fillna(0)
    values.append((sum((df.b-df.coef)**2))**0.5)
    
plt.plot(np.arange(1,21), values)
plt.scatter(values.index(min(values))+1, min(values), s=100, c='r')
plt.xticks(np.arange(1, 21, step=1))
# Plot is very similar to test MSE plot in (d).

# ----------------------------------------------------------------------------
# 11
# (a)
Boston = pd.read_csv('C:\\Users\\Carol\\Desktop\\Boston.csv')

x = Boston.iloc[:,2:]
y = Boston['crim']

np.random.seed(1)
train = np.random.choice([True, False], size = len(y), replace = True)
test  = np.invert(train)

# Use forward stepwise selection to save run time
np.random.seed(2)
folds = np.random.choice(10, size=len(y[train]), replace = True) # 10-fold CV

cv_xi  = pd.DataFrame(columns=range(1,11), index=range(1,len(x.columns)+1))
cv_err = pd.DataFrame(columns=range(1,11), index=range(1,len(x.columns)+1))

for k in range(1, 11):    
    
    cv_ytest = y[train][folds == k-1]
    cv_xtest = x[train][folds == k-1] 
    
    cv_ytrain = y[train][folds != k-1]
    cv_xtrain = x[train][folds != k-1] 
    
    oldxs   = []
    remainx = cv_xtrain
    
    for i in range(1, len(x.columns)+1):       
        mse      = []
        xcombo   = []
        best_mse = np.inf

        for combo in itertools.combinations(remainx.columns, 1):
            tempx1 = sm.add_constant(cv_xtrain[oldxs + list(combo)])
            lm     = sm.OLS(cv_ytrain, tempx1).fit()
            
            tempx2   = sm.add_constant(cv_xtest[tempx1.columns[1:]])
            test_mse = mean_squared_error(cv_ytest, lm.predict(tempx2))           
            mse.append(test_mse)
            xcombo.append(combo)
            
        if min(mse) < best_mse:
            addx = xcombo[mse.index(min(mse))][0]
            # a list includes all the x's that have been chosen
            # this updates the list, so later we must make a copy for recording
            oldxs.append(addx)
            # remove selected x's from the dataset
            remainx = cv_xtrain.drop(labels=list(oldxs), axis=1)
            # record info 
            best_mse = min(mse)
            best_xi  = oldxs.copy()
            
        cv_xi[k][i]  = best_xi
        cv_err[k][i] = best_mse
        
# best model by # of features
cv_err_mean = cv_err.apply(np.mean, axis=1)
plt.plot(cv_err_mean)
plt.xlabel('# of features')
plt.ylabel('CV MSE')
plt.scatter(cv_err_mean.argmin(), cv_err_mean.min(), s=100, c='r') #7

# compute MSE using test data
tempx  = sm.add_constant(x[test][cv_xi[cv_err.loc[cv_err_mean.argmin(),:].argmin()][cv_err_mean.argmin()]])
lm     = sm.OLS(y[test],tempx).fit()
lm_err = mean_squared_error(y[test], lm.predict(tempx)) #30.39

# LASSO
lassocv    = LassoCV(cv=10, normalize=True).fit(x[train], y[train]) # 10-fold CV
lasso_pred = lassocv.predict(x[test])
lasso_err  = mean_squared_error(y[test], lasso_pred) #33.40
lasso_coef = pd.Series(lassocv.coef_, index = x[train].columns)
print(lasso_coef.abs().sort_values(ascending=False)) # rad, dis, medv, black
print('# of non-zero coef estimates: %i' %lasso_coef[lasso_coef != 0].count()) #4
      
# Ridge
alphas     = 10**np.linspace(10,-2,100)*0.5
ridgecv    = RidgeCV(alphas=alphas, cv=10, normalize = True).fit(x[train], y[train]) # 10-fold CV
ridge_pred = ridgecv.predict(x[test])
ridge_err  = mean_squared_error(y[test], ridge_pred) #35.18
ridge_coef = pd.Series(ridgecv.coef_, index = x[train].columns)
print(ridge_coef.abs().sort_values(ascending=False)) 

# PCR
# scale data
pca = PCA()
reduced_xtrain = pca.fit_transform(scale(x[train]))
# amount of variance explained by adding each principal component
np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

# LOOCV with shuffle
ntrain = len(reduced_xtrain)
kf10   = KFold(n_splits=10, shuffle=True, random_state=1)

# pick M based on MSE
pcr    = LinearRegression()
mse    = []
# compute MSE with only intercept (no principal components)
score = -1*cross_val_score(pcr, np.ones((ntrain,1)), y[train].ravel(), cv=kf10, scoring='neg_mean_squared_error').mean()    
mse.append(score)
# compute MSE with the 13 principle components, adding one at a time
for i in np.arange(1,14):
    score = -1*cross_val_score(pcr, reduced_xtrain[:,:i], y[train].ravel(), cv=kf10, scoring='neg_mean_squared_error').mean()
    mse.append(score)
# plot MSE against # of components
plt.plot(mse, '-o')
plt.xlabel('# of principal components')
plt.ylabel('MSE')
# select M
M = mse.index(min(mse)) # index is 0 for intercept-only MSE
print('M = %i' %M) #13

# predict using M and test data
reduced_xtest = pca.transform(scale(x[test]))[:,:M+1]

pcr.fit(reduced_xtrain[:,:M+1], y[train])
pcr_pred = pcr.predict(reduced_xtest)
pcr_err  = mean_squared_error(y[test], pcr_pred) #32.36

# (b)
err = pd.DataFrame({'errors': [lm_err, ridge_err, lasso_err, pcr_err]})
err.rename(index={0:'fwd', 1:'ridge', 2:'lasso', 3:'pcr'}, inplace=True) 
print(err)

x_label = np.arange(4)
plt.bar(x_label, err.errors)
plt.xticks(x_label, ('fwd', 'ridge', 'lasso', 'pcr'))
plt.ylabel('test MSE')
# Best subset forward performs the best and Ridge the worst.

# (c) No, because some features hardly improve our prediction accuracy.