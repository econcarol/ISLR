# ISLR Ch 7 by Carol Cui

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from patsy import dmatrix
from pygam import LinearGAM, l, s, f

# ----------------------------------------------------------------------------
# Q6
Wage = pd.read_csv('C:\\Users\\Carol\\Desktop\\Wage.csv')

# (a)
# CV
lm     = LinearRegression()
cv_mse = np.zeros(10)

for i in range(1,11):
    age_poly    = PolynomialFeatures(i).fit_transform(Wage['age'].values.reshape(-1,1))
    lm_fit      = lm.fit(age_poly, Wage['wage'])
    kf10        = KFold(n_splits=10, random_state=1)
    scores      = cross_val_score(lm_fit, age_poly, Wage['wage'], scoring="neg_mean_squared_error", cv=kf10)
    cv_mse[i-1] = np.mean(np.abs(scores))
    
plt.plot(range(1,11), cv_mse)
plt.scatter(cv_mse.argmin()+1, cv_mse.min(), s=100, c='r')
plt.xlabel('# of polynomial degrees')
plt.ylabel('10-fold CV MSE')
# 4 is chosen by 10-fold CV.

# ANOVA
fit = []
for i in range(1,11):
    age_poly = PolynomialFeatures(i).fit_transform(Wage['age'].values.reshape(-1,1))
    lm_fit   = sm.OLS(Wage['wage'], age_poly).fit()
    fit.append(lm_fit) 
  
print(sm.stats.anova_lm(fit[0], fit[1], fit[2], fit[3], fit[4], fit[5], fit[6], fit[7], fit[8], fit[9], typ=1))
# 4 is chosen by anova.

# plot
age_grid = np.arange(Wage['age'].min(), Wage['age'].max()).reshape(-1,1)
x_test   = PolynomialFeatures(4).fit_transform(age_grid)
y_pred   = fit[3].predict(x_test)
se_pred  = fit[3].get_prediction(x_test).summary_frame()['mean_se']

fig, ax = plt.subplots(1, 1)
fig.suptitle('Degree-4 Polynomial')
ax.scatter(Wage['age'], Wage['wage'], facecolor='None', edgecolor='k', alpha=0.3)
ax.plot(age_grid, y_pred, color='b')
ax.plot(age_grid, y_pred+2*se_pred, color='b', linestyle='dashed')
ax.plot(age_grid, y_pred-2*se_pred, color='b', linestyle='dashed')
ax.set_xlabel('age')
ax.set_ylabel('wage')

# (b)
lm     = LinearRegression()
cv_mse = np.zeros(9)

for i in range(2,11):
    age_cut, bins  = pd.cut(Wage['age'], i, retbins=True, right=True)
    age_cut.value_counts(sort=False)
    age_step       = pd.concat([Wage['age'], age_cut, Wage['wage']], keys=['age', 'age_cut', 'wage'], axis = 1)
    age_step_dummy = pd.get_dummies(age_step['age_cut'])
    age_step_dummy = age_step_dummy.drop(age_step_dummy.columns[0], axis = 1)

    lm_fit      = lm.fit(age_step_dummy, Wage['wage'])
    kf10        = KFold(n_splits=10, random_state=1)
    scores      = cross_val_score(lm_fit, age_step_dummy, Wage['wage'], scoring="neg_mean_squared_error", cv=kf10)
    cv_mse[i-2] = np.mean(np.abs(scores))
    
plt.plot(range(2,11), cv_mse)
plt.scatter(cv_mse.argmin()+2, cv_mse.min(), s=100, c='r')
plt.xlabel('# of age bins')
plt.ylabel('10-fold CV MSE')
# 8 is chosen by 10-fold CV.

# plot
age_cut, bins  = pd.cut(Wage['age'], cv_mse.argmin()+2, retbins=True, right=True)
age_cut.value_counts(sort=False)  
age_step       = pd.concat([Wage['age'], age_cut, Wage['wage']], keys=['age', 'age_cut', 'wage'], axis = 1)
age_step_dummy = pd.get_dummies(age_step['age_cut'])
age_step_dummy = age_step_dummy.drop(age_step_dummy.columns[0], axis = 1)
age_step_dummy = sm.add_constant(age_step_dummy)

# use statsmodels as it reports prediction SE
lm_fit = sm.OLS(age_step.wage, age_step_dummy).fit()
    
age_grid    = np.arange(Wage['age'].min(), Wage['age'].max()).reshape(-1,1)
bin_mapping = np.digitize(age_grid.ravel(), bins)
x_test      = sm.add_constant(pd.get_dummies(bin_mapping).drop(1, axis = 1))
y_pred      = lm_fit.predict(x_test)
se_pred     = lm_fit.get_prediction(x_test).summary_frame()['mean_se']

fig, ax = plt.subplots(1, 1)
fig.suptitle('8-Age-Bin Step Function')
ax.scatter(Wage['age'], Wage['wage'], facecolor='None', edgecolor='k', alpha=0.3)
ax.plot(age_grid, y_pred, color='b')
ax.plot(age_grid, y_pred+2*se_pred, color='b', linestyle='dashed')
ax.plot(age_grid, y_pred-2*se_pred, color='b', linestyle='dashed')
ax.set_xlabel('age')
ax.set_ylabel('wage')

# ----------------------------------------------------------------------------
# Q7
Wage.boxplot(column=['wage'], by=['maritl'])
# Wage is highest for the married. 
Wage.boxplot(column=['wage'], by=['jobclass'])
# Wage is higher for information job class.

mar_dummy = pd.get_dummies(Wage['maritl'])
mar_dummy = mar_dummy.drop(mar_dummy.columns[0], axis = 1)
job_dummy = pd.get_dummies(Wage['jobclass'])
job_dummy = job_dummy.drop(job_dummy.columns[0], axis = 1)
age_sp4   = dmatrix("bs(Wage.age, df=4, include_intercept=False)", {"Wage.age": Wage.age}, return_type='dataframe')

x1   = pd.concat([age_sp4, mar_dummy], axis = 1)
fit1 = sm.OLS(Wage['wage'], x1).fit()
x2   = pd.concat([age_sp4, job_dummy], axis = 1)
fit2 = sm.OLS(Wage['wage'], x2).fit()
x3   = pd.concat([age_sp4, mar_dummy, job_dummy], axis = 1)
fit3 = sm.OLS(Wage['wage'], x3).fit()
print(sm.stats.anova_lm(fit1, fit2, fit3, typ=1))
# Both marital status and job class should be included. 

# plot
def marital_d(row):
    if row['maritl'] == '1. Never Married':
        val = 0
    elif row['maritl'] == '2. Married':
        val = 1
    elif row['maritl'] == '3. Widowed':
        val = 2
    elif row['maritl'] == '4. Divorced':
        val = 3
    else:
        val = 4
    return val
Wage['mar_d'] = Wage.apply(marital_d, axis=1)
Wage['job_d'] = np.where(Wage['jobclass']=='1. Industrial', 0, 1)

x3  = pd.concat([Wage['age'], Wage['mar_d'], Wage['job_d']], axis = 1)
gam = LinearGAM(s(0,n_splines=4) + f(1) + f(2)).fit(x3, Wage['wage'])

for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue

    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

    plt.figure()
    plt.plot(XX[:, term.feature], pdep)
    plt.plot(XX[:, term.feature], confi, c='r', ls='--')
    plt.title(repr(term))
    plt.show()

# ----------------------------------------------------------------------------
# Q8
Auto = pd.read_csv('C:\\Users\\Carol\\Desktop\\Auto.csv', na_values='?') 
Auto = Auto.dropna() # drop all missing values

pd.tools.plotting.scatter_matrix(Auto, figsize=(10,10))
# non-linear relationship w/ mpg: displacement, horsepower, weight

# select polynomial degrees by CV
lm     = LinearRegression()
ds_mse = np.zeros(10)
for i in range(1,11):
    poly        = PolynomialFeatures(i).fit_transform(Auto['displacement'].values.reshape(-1,1))
    lm_fit      = lm.fit(poly, Auto['mpg'])
    kf10        = KFold(n_splits=10, random_state=1)
    scores      = cross_val_score(lm_fit, poly, Auto['mpg'], scoring="neg_mean_squared_error", cv=kf10)
    ds_mse[i-1] = np.mean(np.abs(scores))
    
hr_mse = np.zeros(10)
for i in range(1,11):
    poly        = PolynomialFeatures(i).fit_transform(Auto['horsepower'].values.reshape(-1,1))
    lm_fit      = lm.fit(poly, Auto['mpg'])
    kf10        = KFold(n_splits=10, random_state=1)
    scores      = cross_val_score(lm_fit, poly, Auto['mpg'], scoring="neg_mean_squared_error", cv=kf10)
    hr_mse[i-1] = np.mean(np.abs(scores))
    
wt_mse = np.zeros(10)
for i in range(1,11):
    poly        = PolynomialFeatures(i).fit_transform(Auto['weight'].values.reshape(-1,1))
    lm_fit      = lm.fit(poly, Auto['mpg'])
    kf10        = KFold(n_splits=10, random_state=1)
    scores      = cross_val_score(lm_fit, poly, Auto['mpg'], scoring="neg_mean_squared_error", cv=kf10)
    wt_mse[i-1] = np.mean(np.abs(scores))
    
fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
fig.suptitle('best # of polynomial degrees')
ax1.plot(range(1,11), ds_mse)
ax1.scatter(ds_mse.argmin()+1, ds_mse.min(), s=100, c='r') #2
ax1.set_title('displacement')
ax1.set_ylabel('10-fold CV MSE')
ax2.plot(range(1,11), hr_mse)
ax2.scatter(hr_mse.argmin()+1, hr_mse.min(), s=100, c='r') #6
ax2.set_title('horsepower')
ax2.set_ylabel('10-fold CV MSE')
ax3.plot(range(1,11), wt_mse)
ax3.scatter(wt_mse.argmin()+1, wt_mse.min(), s=100, c='r') #2
ax3.set_title('weight')
ax3.set_ylabel('10-fold CV MSE')

ds_poly = PolynomialFeatures(2).fit_transform(Auto['displacement'].values.reshape(-1,1))
hr_poly = PolynomialFeatures(6).fit_transform(Auto['horsepower'].values.reshape(-1,1))
wt_poly = PolynomialFeatures(2).fit_transform(Auto['weight'].values.reshape(-1,1))

x1   = hr_poly
fit1 = sm.OLS(Auto['mpg'], x1).fit()
x2   = np.concatenate([hr_poly, np.delete(ds_poly,0,1)], axis = 1)
fit2 = sm.OLS(Auto['mpg'], x2).fit()
x3   = np.concatenate([hr_poly, np.delete(wt_poly,0,1)], axis = 1)
fit3 = sm.OLS(Auto['mpg'], x3).fit()
x4   = np.concatenate([hr_poly, np.delete(ds_poly,0,1), np.delete(wt_poly,0,1)], axis = 1)
fit4 = sm.OLS(Auto['mpg'], x4).fit()
print(sm.stats.anova_lm(fit1, fit2, fit3, fit4, typ=1))
# All 3 should be included. 

# ----------------------------------------------------------------------------
# Q9
Boston = pd.read_csv('C:\\Users\\Carol\\Desktop\\Boston.csv')

# (a)
# regression results
dis3 = PolynomialFeatures(3).fit_transform(Boston['dis'].values.reshape(-1,1))
fit3 = sm.OLS(Boston['nox'], dis3).fit()
print(fit3.summary())

# plot
dis_grid = np.arange(Boston['dis'].min(), Boston['dis'].max()).reshape(-1,1)
x_test   = PolynomialFeatures(3).fit_transform(dis_grid)
y_pred   = fit3.predict(x_test)
se_pred  = fit3.get_prediction(x_test).summary_frame()['mean_se']

fig, ax = plt.subplots(1, 1)
fig.suptitle('Degree-3 Polynomial')
ax.scatter(Boston['dis'], Boston['nox'], facecolor='None', edgecolor='k', alpha=0.3)
ax.plot(dis_grid, y_pred, color='b')
ax.plot(dis_grid, y_pred+2*se_pred, color='b', linestyle='dashed')
ax.plot(dis_grid, y_pred-2*se_pred, color='b', linestyle='dashed')
ax.set_xlabel('distance')
ax.set_ylabel('nitrogen oxides concentration')

# (b)
rss = np.zeros(10)
for i in range(1,11):
    poly     = PolynomialFeatures(i).fit_transform(Boston['dis'].values.reshape(-1,1))
    fit      = sm.OLS(Boston['nox'], poly).fit()
    rss[i-1] = sum(fit.resid**2)
plt.plot(range(1,11), rss)
plt.scatter(rss.argmin()+1, min(rss), color='r')
plt.xlabel('# of polynomial degrees')
plt.ylabel('RSS')
# 10 is chosen.

# (c)
lm     = LinearRegression()
cv_mse = np.zeros(10)
for i in range(1,11):
    poly        = PolynomialFeatures(i).fit_transform(Boston['dis'].values.reshape(-1,1))
    lm_fit      = lm.fit(poly, Boston['nox'])
    kf10        = KFold(n_splits=10, random_state=1)
    scores      = cross_val_score(lm_fit, poly, Boston['nox'], scoring="neg_mean_squared_error", cv=kf10)
    cv_mse[i-1] = np.mean(np.abs(scores))
    
plt.plot(range(1,11), cv_mse)
plt.scatter(cv_mse.argmin()+1, cv_mse.min(), s=100, c='r')
plt.xlabel('# of polynomial degrees')
plt.ylabel('10-fold CV MSE')
# 3 is chosen by 10-fold CV.

# (d)
dis_sp4 = dmatrix("bs(Boston.dis, df=4, include_intercept=False)", {"Boston.dis": Boston.dis}, return_type='dataframe')
fit     = sm.OLS(Boston['nox'], dis_sp4).fit()
x_test  = dmatrix("bs(dis_grid, df=4, include_intercept=False)", {"dis_grid": dis_grid}, return_type='dataframe')
y_pred  = fit.predict(x_test) 
se_pred = fit.get_prediction(x_test).summary_frame()['mean_se']

fig, ax = plt.subplots(1, 1)
fig.suptitle('nox fitted w/ 4-df spline of dis')
ax.scatter(Boston['dis'], Boston['nox'], facecolor='None', edgecolor='k', alpha=0.3)
ax.plot(dis_grid, y_pred, color='b')
ax.plot(dis_grid, y_pred+2*se_pred, color='b', linestyle='dashed')
ax.plot(dis_grid, y_pred-2*se_pred, color='b', linestyle='dashed')
ax.set_xlabel('distance')
ax.set_ylabel('nitrogen oxides concentration')

# (e)
rss = np.zeros(7)
for i in range(4,11):
    sp       = dmatrix("bs(Boston.dis, df=i, include_intercept=False)", {"Boston.dis": Boston.dis}, return_type='dataframe')
    fit      = sm.OLS(Boston['nox'], sp).fit()
    rss[i-4] = sum(fit.resid**2)

plt.plot(range(4,11), rss)
plt.xlabel('degree of freedom for spline')
plt.ylabel('RSS')
# RSS decreases in df.

# (f)
fits = []
for i in range(4,11):
    sp       = dmatrix("bs(Boston.dis, df=i, include_intercept=False)", {"Boston.dis": Boston.dis}, return_type='dataframe')
    fits.append(sm.OLS(Boston['nox'], sp).fit())
    
print(sm.stats.anova_lm(fits[0], fits[1], fits[2], fits[3], fits[4], fits[5], fits[6], typ=1))
# 2, 5 and 7 are the better choices.

# ----------------------------------------------------------------------------
# Q10
College     = pd.read_csv('C:\\Users\\Carol\\Desktop\\College.csv')
school_name = College.iloc[:,0] 
College     = College.rename(index = lambda x: school_name[x]) 
College.drop(College.columns[[0]], axis=1, inplace=True)

College['Private01'] = np.where(College.Private=='Yes', 1, 0)
College.drop(['Private'], axis = 1, inplace=True)

# (a)
X = College.drop(['Outstate'], axis=1).astype('float64')
Y = College['Outstate']
xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.5, random_state=1)

# forward stepwise selection
oldxs  = []
xcombo = []
coef   = []
adjr2  = []
bic    = []
mse    = []
ssr    = []

p       = len(xtrain.columns)
remainx = xtrain

for i in range(1, p+1):
       
    best_ssr = np.inf
    
    for combo in itertools.combinations(remainx.columns,1):
        tempx = sm.add_constant(xtrain[oldxs + list(combo)])
        lm    = sm.OLS(ytrain,tempx).fit()
        
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
    remainx = xtrain.drop(labels=list(oldxs), axis=1)
        
    # record info
    xcombo.append(oldxs.copy())
    coef.append(best_coef)    
    adjr2.append(best_adjr2)
    bic.append(best_bic)
    mse.append(best_mse)
    ssr.append(best_ssr)

results = pd.DataFrame({'xnum':range(1,p+1), 'xcombo':xcombo, 'coef':coef,
                        'adj_R2':adjr2, 'BIC':bic, 'MSE':mse, 'RSS':ssr})

# adjusted R2
max_adjr2 = results.adj_R2.max()
plt.plot(range(1,p+1), results['adj_R2'])
plt.scatter(results.adj_R2.argmax()+1, max_adjr2, s=100, c='r') #15
plt.xlabel('# of features')
plt.ylabel('adjusted R-squared')

best_adjr2_xcombo = results[results['adj_R2']==max_adjr2].xcombo
print(*best_adjr2_xcombo)
best_adjr2_coef   = results[results['adj_R2']==max_adjr2].coef
print(best_adjr2_coef.values.reshape(-1,1))

# BIC
min_bic = results.BIC.min()
plt.plot(range(1,p+1), results['BIC'])
plt.scatter(results.BIC.argmin()+1, min_bic, s=100, c='r') #9
plt.xlabel('# of features')
plt.ylabel('BIC')

best_bic_xcombo = results[results['BIC']==min_bic].xcombo
print(*best_bic_xcombo)
best_bic_coef   = results[results['BIC']==min_bic].coef
print(best_bic_coef.values.reshape(-1,1))

# Cp
results['MSE_all'] = sm.OLS(ytrain,sm.add_constant(xtrain)).fit().mse_resid
results['Cp']      = results['xnum'] + (results['MSE']-results['MSE_all'])*(len(xtrain)-results['xnum'])/results['MSE_all']

min_cp = results.Cp.min()
plt.plot(range(1,p+1), results['Cp'])
plt.scatter(results.Cp.argmin()+1, min_cp, s=100, c='r') #12
plt.xlabel('# of features')
plt.ylabel('Cp')

best_cp_xcombo = results[results['Cp']==min_cp].xcombo
print(*best_cp_xcombo)
best_cp_coef   = results[results['Cp']==min_cp].coef
print(best_cp_coef.values.reshape(-1,1))
# Room.Board, perc.alumni, Expend, Private01, PhD, Grad.Rate, Personal, Accept, Enroll

# (b)
fig, axs = plt.subplots(3, 3)
axs[0,0].scatter(xtrain['Room.Board'], ytrain)  # seems linear
axs[0,0].set_title('Room.Board')
axs[0,1].scatter(xtrain['perc.alumni'], ytrain) # seems linear
axs[0,1].set_title('perc.alumni')
axs[0,2].scatter(xtrain['Expend'], ytrain)
axs[0,2].set_title('Expend')
axs[1,0].scatter(xtrain['Private01'], ytrain)
axs[1,0].set_title('Private01')
axs[1,1].scatter(xtrain['PhD'], ytrain)
axs[1,1].set_title('PhD')
axs[1,2].scatter(xtrain['Grad.Rate'], ytrain)   # seems linear
axs[1,2].set_title('Grad.Rate')
axs[2,0].scatter(xtrain['Personal'], ytrain)
axs[2,0].set_title('Personal')
axs[2,1].scatter(xtrain['Accept'], ytrain)
axs[2,1].set_title('Accept')
axs[2,2].scatter(xtrain['Enroll'], ytrain)
axs[2,2].set_title('Enroll')

sp1 = dmatrix("bs(xtrain['Room.Board'], df=6, include_intercept=False)", {"xtrain['Room.Board']": xtrain['Room.Board']}, return_type='dataframe')
sp2 = dmatrix("bs(xtrain['perc.alumni'], df=6, include_intercept=False)", {"xtrain['perc.alumni']": xtrain['perc.alumni']}, return_type='dataframe')
sp3 = dmatrix("bs(xtrain['Expend'], df=6, include_intercept=False)", {"xtrain['Expend']": xtrain['Expend']}, return_type='dataframe')
sp4 = dmatrix("bs(xtrain['PhD'], df=6, include_intercept=False)", {"xtrain['PhD']": xtrain['PhD']}, return_type='dataframe')
sp5 = dmatrix("bs(xtrain['Grad.Rate'], df=6, include_intercept=False)", {"xtrain['Grad.Rate']": xtrain['Grad.Rate']}, return_type='dataframe')
sp6 = dmatrix("bs(xtrain['Personal'], df=6, include_intercept=False)", {"xtrain['Personal']": xtrain['Personal']}, return_type='dataframe')
sp7 = dmatrix("bs(xtrain['Accept'], df=6, include_intercept=False)", {"xtrain['Accept']": xtrain['Accept']}, return_type='dataframe')
sp8 = dmatrix("bs(xtrain['Enroll'], df=6, include_intercept=False)", {"xtrain['Enroll']": xtrain['Enroll']}, return_type='dataframe')

x1   = xtrain[list(['Room.Board', 'perc.alumni', 'Expend', 'Private01', 'PhD', 'Grad.Rate', 'Personal', 'Accept', 'Enroll'])]
fit1 = sm.OLS(ytrain,sm.add_constant(x1)).fit()

x2   = pd.concat([xtrain['Private01'], xtrain['Room.Board'], xtrain['perc.alumni'], xtrain['Grad.Rate'], sp3, sp4, sp6, sp7, sp8], axis = 1)
fit2 = sm.OLS(ytrain,sm.add_constant(x2)).fit()

x3   = pd.concat([xtrain['Private01'], xtrain['Room.Board'], xtrain['perc.alumni'], sp5, sp3, sp4, sp6, sp7, sp8], axis = 1)
fit3 = sm.OLS(ytrain,sm.add_constant(x3)).fit()

x4   = pd.concat([xtrain['Private01'], xtrain['Room.Board'], sp2, xtrain['Grad.Rate'], sp3, sp4, sp6, sp7, sp8], axis = 1)
fit4 = sm.OLS(ytrain,sm.add_constant(x4)).fit()

x5   = pd.concat([xtrain['Private01'], sp1, xtrain['perc.alumni'], xtrain['Grad.Rate'], sp3, sp4, sp6, sp7, sp8], axis = 1)
fit5 = sm.OLS(ytrain,sm.add_constant(x5)).fit()

x6   = pd.concat([xtrain['Private01'], xtrain['Room.Board'], sp2, sp5, sp3, sp4, sp6, sp7, sp8], axis = 1)
fit6 = sm.OLS(ytrain,sm.add_constant(x6)).fit()

x7   = pd.concat([xtrain['Private01'], sp1, xtrain['perc.alumni'], sp5, sp3, sp4, sp6, sp7, sp8], axis = 1)
fit7 = sm.OLS(ytrain,sm.add_constant(x7)).fit()

x8   = pd.concat([xtrain['Private01'], sp1, sp2, xtrain['Grad.Rate'], sp3, sp4, sp6, sp7, sp8], axis = 1)
fit8 = sm.OLS(ytrain,sm.add_constant(x8)).fit()

x9   = pd.concat([xtrain['Private01'], sp1, sp2, sp5, sp3, sp4, sp6, sp7, sp8], axis = 1)
fit9 = sm.OLS(ytrain,sm.add_constant(x9)).fit()

print(sm.stats.anova_lm(fit1, fit2, fit3, fit4, fit5, fit6, fit7, fit8, fit9, typ=1))
# 2 is better. 

# (c)
# forward stepwise
x1test = xtest[list(['Room.Board', 'perc.alumni', 'Expend', 'Private01', 'PhD', 'Grad.Rate', 'Personal', 'Accept', 'Enroll'])]
ypred  = fit1.predict(sm.add_constant(x1test))
print('fwd stepwise MSE: %.2f' %mean_squared_error(ytest, ypred)) #4297712.56

# selected GAM
sp3t = dmatrix("bs(xtest['Expend'], df=6, include_intercept=False)", {"xtest['Expend']": xtest['Expend']}, return_type='dataframe')
sp4t = dmatrix("bs(xtest['PhD'], df=6, include_intercept=False)", {"xtest['PhD']": xtest['PhD']}, return_type='dataframe')
sp6t = dmatrix("bs(xtest['Personal'], df=6, include_intercept=False)", {"xtest['Personal']": xtest['Personal']}, return_type='dataframe')
sp7t = dmatrix("bs(xtest['Accept'], df=6, include_intercept=False)", {"xtest['Accept']": xtest['Accept']}, return_type='dataframe')
sp8t = dmatrix("bs(xtest['Enroll'], df=6, include_intercept=False)", {"xtest['Enroll']": xtest['Enroll']}, return_type='dataframe')

x2test = pd.concat([xtest['Private01'], xtest['Room.Board'], xtest['perc.alumni'], xtest['Grad.Rate'], sp3t, sp4t, sp6t, sp7t, sp8t], axis = 1)
ypred  = fit2.predict(sm.add_constant(x2test))
print('GAM MSE: %.2f' %mean_squared_error(ytest, ypred)) #3876188.82

# (d)
gam = LinearGAM(l(0) + l(1) + s(2,n_splines=6) + f(3)+ s(4,n_splines=6) + l(5) + s(6,n_splines=6) + s(7,n_splines=6) + s(8,n_splines=6)).fit(x1, ytrain)

for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue

    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

    plt.figure()
    plt.plot(XX[:, term.feature], pdep)
    plt.plot(XX[:, term.feature], confi, c='r', ls='--')
    plt.title(repr(term))
    plt.show()
# Non-linear: Expand, Personal, Accept. 
    
# ----------------------------------------------------------------------------
# Q11
# (a)
np.random.seed(1234)
n  = 1000
x1 = np.random.randn(n)
x2 = np.random.randn(n)
e  = np.random.randn(n)
beta0 = 0.5
beta1 = 2.6
beta2 = 5.3
y  = beta0 + beta1*x1 + beta2*x2 + e

# (b)
beta1 = 4

# (c)
a     = y - beta1*x1
lm    = LinearRegression()
beta2 = lm.fit(x2.reshape(-1,1), a).coef_[0]
print(beta2)

# (d)
a     = y - beta2*x2
beta1 = lm.fit(x1.reshape(-1,1), a).coef_[0]
print(beta1)

# (e)
all_beta0 = np.zeros(1000)
all_beta1 = np.zeros(1000)
all_beta2 = np.zeros(1000)

for i in range(0,1000):
    a = y - all_beta1[i]*x1
    all_beta2[i] = lm.fit(x2.reshape(-1,1), a).coef_[0]
    
    a = y - all_beta2[i]*x2
    all_beta1[i] = lm.fit(x1.reshape(-1,1), a).coef_[0]
    all_beta0[i] = lm.fit(x1.reshape(-1,1), a).intercept_

plt.scatter(range(1,1001), all_beta0, color='b')
plt.scatter(range(1,1001), all_beta1, color='r')
plt.scatter(range(1,1001), all_beta2, color='g')
plt.legend(('beta0', 'beta1', 'beta2'), loc='best')
plt.xlabel('nth iteration')
plt.ylabel('beta estimates')
plt.ylim((0,6))

# (f)
x = pd.DataFrame({'x1':x1, 'x2':x2})
lm_beta0 = lm.fit(x,y).intercept_
lm_beta1 = lm.fit(x,y).coef_[0]
lm_beta2 = lm.fit(x,y).coef_[1]

plt.axhline(y=lm_beta0, color='orange', linestyle='dashed')
plt.axhline(y=lm_beta1, color='k', linestyle='dashed')
plt.axhline(y=lm_beta2, color='gray', linestyle='dashed')
plt.scatter(range(1,1001), all_beta0, color='b')
plt.scatter(range(1,1001), all_beta1, color='r')
plt.scatter(range(1,1001), all_beta2, color='g')
plt.legend(('lm_beta0', 'lm_beta1', 'lm_beta2', 'beta0', 'beta1', 'beta2'), loc='best')
plt.xlabel('nth iteration')
plt.ylabel('beta estimates')
plt.ylim((0,6))

# (g) One iteration is good enough based on graph in (f).

# ----------------------------------------------------------------------------
# Q12
# (a)
np.random.seed(1234)
n = 1000                 # # of obs
p = 100                  # # of x's
c = np.random.randn(1)   # intercept
b = np.random.randn(p)   # coefficients
e = np.random.randn(n)   # error term
x = np.random.randn(n,p)
y = c + np.matmul(x,b) + e

# multiple linear regression results
lm      = LinearRegression()
lm_bhat = np.zeros(p)
for j in range(0,p):
    lm_bhat[j] = lm.fit(x, y).coef_[j]

# backfitting results
ite  = 100 # # of iterations
mse  = np.zeros(ite)
dif  = np.zeros(ite)
chat = np.zeros((ite, p))
bhat = np.zeros((ite, p))

for i in range(0,ite):
    for j in range(0,p):
        tempx = np.concatenate([x[:,:j], x[:,j+1:]], axis=1)
        tempb = np.delete(bhat[i,:], j, 0)
        a     = y - np.matmul(tempx,tempb)
        
        chat[i:ite,j] = lm.fit(x[:,j].reshape(-1,1), a).intercept_ 
        bhat[i:ite,j] = lm.fit(x[:,j].reshape(-1,1), a).coef_[0]
        
    mse[i] = mean_squared_error(y, chat[i,j]+np.matmul(x,bhat[i,:]))
    dif[i] = mean_squared_error(lm_bhat, bhat[i,:])

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.scatter(range(1,ite+1), mse)
ax1.set_xlabel('nth iteration')
ax1.set_ylabel('backfitting MSE for y') 
ax1.set_ylim(ymin=min(mse)-0.5, ymax=max(mse)+0.5)
ax2.scatter(range(1,ite+1), dif)
ax2.set_xlabel('nth iteration')
ax2.set_ylabel('lm vs. backfitting MSE for betas') 
ax2.set_ylim(ymin=min(dif)-0.001, ymax=max(dif)+0.001)
# 3 ieration is good enough based on both graphs. 
