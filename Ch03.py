# ISLR Ch 3 by Carol Cui
%reset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from   statsmodels.stats.api import anova_lm

# ----------------------------------------------------------------------------
# Q8
# import data: treat ? as missing value
Auto = pd.read_csv('C:\\Users\\Carol\\Desktop\\Auto.csv', na_values='?') 
Auto = Auto.dropna() # drop all missing values
pd.options.display.max_rows = 10 # set max # of rows for display
Auto # show data

# (a)
# reshape data
n = len(Auto['mpg']) # total # of obs
X = np.reshape(Auto['horsepower'], (n,1))
Y = np.reshape(Auto['mpg'], (n,1))
# run regression
X          = sm.add_constant(X) # add intercept
model1     = sm.OLS(Y, X)
model1_fit = model1.fit()
print(model1_fit.summary())

# (i): Yes.
# (ii): Strong as p-value is very close to 0.
# (iii): Negative.
# (iv): 
xnew = np.array([[1., 98.]])
ynew = model1_fit.get_prediction(xnew)
print(ynew.summary_frame(alpha=0.05))

# (b)
a    = model1_fit.params[0]
b    = model1_fit.params[1]
yfit = [a + b*xi for xi in X[:,1]]

plt.scatter(X[:, 1], Y)
plt.xlabel('horsepower')
plt.ylabel('mpg')
plt.plot(X[:,1], yfit, 'red')

# (c)
# fitted values
model1_fitted_y = model1_fit.fittedvalues
# normalized residuals
model1_norm_residuals = model1_fit.get_influence().resid_studentized_internal
# absolute squared normalized residuals
model1_norm_residuals_abs_sqrt = np.sqrt(np.abs(model1_norm_residuals))
# leverage, from statsmodels internals
model1_leverage = model1_fit.get_influence().hat_matrix_diag

# plot 1: residuals vs. fitted values
plot1 = plt.figure(1)
sns.residplot(model1_fitted_y, Auto['mpg'], lowess=True, line_kws={'color':'red', 'lw':1})
plot1.axes[0].set_title('Residuals vs Fitted')
plot1.axes[0].set_xlabel('Fitted values')
plot1.axes[0].set_ylabel('Residuals')

# plot 2: normal Q-Q
plot2 = sm.qqplot(model1_norm_residuals, fit=True, line='45')
plot2.axes[0].set_title('Normal Q-Q')
plot2.axes[0].set_xlabel('Theoretical quantiles')
plot2.axes[0].set_ylabel('Standardized residuals');

# plot 3: scale-location
plot3 = plt.figure(3)
plt.scatter(model1_fitted_y, model1_norm_residuals_abs_sqrt)
sns.regplot(model1_fitted_y, model1_norm_residuals_abs_sqrt, scatter=False, ci=False, lowess=True, line_kws={'color':'red', 'lw':1})
plot3.axes[0].set_title('Scale-Location')
plot3.axes[0].set_xlabel('Fitted values')
plot3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

# plot 4: residuals vs. leverage
plot4 = plt.figure(4)
plt.scatter(model1_leverage, model1_norm_residuals)
sns.regplot(model1_leverage, model1_norm_residuals, scatter=False, ci=False, lowess=True, line_kws={'color':'red', 'lw':1})
plot4.axes[0].set_title('Residuals vs Leverage')
plot4.axes[0].set_xlabel('Leverage')
plot4.axes[0].set_ylabel('Standardized residuals')

# ----------------------------------------------------------------------------
# Q9
# (a)
pd.tools.plotting.scatter_matrix(Auto, figsize=(10,10))

# (b)
Auto.corr()

# (c)
X = np.reshape(Auto.iloc[:,1:8], (n,7))
Y = Auto['mpg'].values.reshape([n,1])
# run regression
X          = sm.add_constant(X) # add intercept
model2     = sm.OLS(Y, X)
model2_fit = model2.fit()
print(model2_fit.summary())

# (i): Yes.
# (ii): displacement, weight, year, origin.
# (iii): Newer models have higher mpg.

# (d)
# fitted values
model2_fitted_y = model2_fit.fittedvalues
# normalized residuals
model2_norm_residuals = model2_fit.get_influence().resid_studentized_internal
# absolute squared normalized residuals
model2_norm_residuals_abs_sqrt = np.sqrt(np.abs(model2_norm_residuals))
# leverage, from statsmodels internals
model2_leverage = model2_fit.get_influence().hat_matrix_diag

# plot 1: residuals vs. fitted values 
plot1 = plt.figure(1)
sns.residplot(model2_fitted_y, Auto['mpg'], lowess=True, line_kws={'color':'red', 'lw':1})
plot1.axes[0].set_title('Residuals vs Fitted')
plot1.axes[0].set_xlabel('Fitted values')
plot1.axes[0].set_ylabel('Residuals')
# Residual plots suggest non-linearity.

# plot 2: normal Q-Q
plot2 = sm.qqplot(model2_norm_residuals, fit=True, line='45')
plot2.axes[0].set_title('Normal Q-Q')
plot2.axes[0].set_xlabel('Theoretical quantiles')
plot2.axes[0].set_ylabel('Standardized residuals');

# plot 3: scale-location
plot3 = plt.figure(3)
plt.scatter(model2_fitted_y, model2_norm_residuals_abs_sqrt)
sns.regplot(model2_fitted_y, model2_norm_residuals_abs_sqrt, scatter=False, ci=False, lowess=True, line_kws={'color':'red', 'lw':1})
plot3.axes[0].set_title('Scale-Location')
plot3.axes[0].set_xlabel('Fitted values')
plot3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

# plot 4: residuals vs. leverage
plot4 = plt.figure(4)
plt.scatter(model2_leverage, model2_norm_residuals)
sns.regplot(model2_leverage, model2_norm_residuals, scatter=False, ci=False, lowess=True, line_kws={'color':'red', 'lw':1})
plot4.axes[0].set_title('Residuals vs Leverage')
plot4.axes[0].set_xlabel('Leverage')
plot4.axes[0].set_ylabel('Standardized residuals')

model2_leverage.argmax()
# Obs 14 has high leverage.

# (e)
model3a = smf.ols(formula='mpg ~ displacement*weight + year', data=Auto) 
print(model3a.fit().summary())
model3b = smf.ols(formula='mpg ~ displacement + weight + origin:year', data=Auto) 
print(model3b.fit().summary())
model3c = smf.ols(formula='mpg ~ displacement + acceleration*weight + year + origin', data=Auto) 
print(model3c.fit().summary())

# (f)
model3d = smf.ols(formula='mpg ~ displacement + np.log(weight) + year', data=Auto) 
print(model3d.fit().summary())
model3e = smf.ols(formula='mpg ~ displacement + np.sqrt(weight) + year + origin', data=Auto) 
print(model3e.fit().summary())
# raw polynomial terms
model3fa = smf.ols(formula='mpg ~ displacement + weight + I(weight**2)', data=Auto) 
print(model3fa.fit().summary())
# orthogonal polynomial terms
def poly(x, p):
    x = np.array(x)
    X = np.transpose(np.vstack((x**k for k in range(p+1))))
    return np.linalg.qr(X)[0][:,1:]

Auto['weight1'] = poly(Auto.weight, 2)[:,0]
Auto['weight2'] = poly(Auto.weight, 2)[:,1]

model3fb = smf.ols(formula='mpg ~ displacement + weight1 + weight2', data=Auto) 
print(model3fb.fit().summary())

# ----------------------------------------------------------------------------
# Q10
Carseats = pd.read_csv('C:\\Users\\Carol\\Desktop\\Carseats.csv') 
Carseats.shape
Carseats.head()

# (a)
model1 = smf.ols(formula='Sales ~ Price + Urban + US', data=Carseats) 
print(model1.fit().summary())
print('RSE = %.3f' %np.sqrt(model1.fit().scale))

# (b)
# Sales drop by 0.054 for each dollar increase in Price.
# Sales are 0.022 lower for Urban areas.
# Sales are 1.201 higher in the US.

# (c)
# Sales = 13.043 - 0.054 x Price - 0.022 x UrbanYes + 1.201 x USYes

# (d)
# Price and USYes

# (e)
model2 = smf.ols(formula='Sales ~ Price + US', data=Carseats) 
print(model2.fit().summary())
print('RSE = %.3f' %np.sqrt(model2.fit().scale))

# (f)
# model1: RSE = 2.472, adj.R2 = 0.234 
# model2: RSE = 2.469, adj.R2 = 0.235

# (g)
model2.fit().conf_int(alpha=0.05)

# (h)
model2_fit = model2.fit()
# fitted values
model2_fitted_y = model2_fit.fittedvalues
# normalized residuals
model2_norm_residuals = model2_fit.get_influence().resid_studentized_internal
# absolute squared normalized residuals
model2_norm_residuals_abs_sqrt = np.sqrt(np.abs(model2_norm_residuals))
# leverage, from statsmodels internals
model2_leverage = model2_fit.get_influence().hat_matrix_diag

# plot 1: residuals vs. fitted values 
plot1 = plt.figure(1)
sns.residplot(model2_fitted_y, Carseats['Sales'], lowess=True, line_kws={'color':'red', 'lw':1})
plot1.axes[0].set_title('Residuals vs Fitted')
plot1.axes[0].set_xlabel('Fitted values')
plot1.axes[0].set_ylabel('Residuals')

# plot 2: normal Q-Q
plot2 = sm.qqplot(model2_norm_residuals, fit=True, line='45')
plot2.axes[0].set_title('Normal Q-Q')
plot2.axes[0].set_xlabel('Theoretical quantiles')
plot2.axes[0].set_ylabel('Standardized residuals');

# plot 3: scale-location
plot3 = plt.figure(3)
plt.scatter(model2_fitted_y, model2_norm_residuals_abs_sqrt)
sns.regplot(model2_fitted_y, model2_norm_residuals_abs_sqrt, scatter=False, ci=False, lowess=True, line_kws={'color':'red', 'lw':1})
plot3.axes[0].set_title('Scale-Location')
plot3.axes[0].set_xlabel('Fitted values')
plot3.axes[0].set_ylabel('$\sqrt{|Standardized Residuals|}$');

# Residual plots look good, so no outliers. 

# plot 4: residuals vs. leverage
plot4 = plt.figure(4)
plt.scatter(model2_leverage, model2_norm_residuals)
sns.regplot(model2_leverage, model2_norm_residuals, scatter=False, ci=False, lowess=True, line_kws={'color':'red', 'lw':1})
plot4.axes[0].set_title('Residuals vs Leverage')
plot4.axes[0].set_xlabel('Leverage')
plot4.axes[0].set_ylabel('Standardized residuals')

Carseats['Unnamed: 0'][model2_leverage.argmax()]
# Data have leverage issues, particularly obs 43.

# ----------------------------------------------------------------------------
# Q11
np.random.seed(1)
x = np.random.normal(size=(100,1))
y = 2*x + np.random.normal(size=(100,1))

# (a)
model1     = sm.OLS(y,x)
model1_fit = model1.fit()
print(model1_fit.summary())

# (b)
model2     = sm.OLS(x,y)
model2_fit = model2.fit()
print(model2_fit.summary())

# (c)
# model 1: y = ax
# model 2: x = by
# Thus, a = 1/b or b = 1/a.

# (d) & (e) skip

# (f)
x2 = sm.add_constant(x) # add intercept
y2 = sm.add_constant(y) # add intercept

model3     = sm.OLS(y,x2)
model3_fit = model3.fit()
print(model3_fit.summary())

model4     = sm.OLS(x,y2)
model4_fit = model4.fit()
print(model4_fit.summary())

# both t = 17.871

# ----------------------------------------------------------------------------
# Q12
# (a) skip
# (b) Q11 is an example
# (c)
np.random.seed(2)
x = np.random.normal(1000, 0.01, size=(100,1))
y = np.random.normal(1000, 0.01, size=(100,1))

model5     = sm.OLS(y,x)
model5_fit = model5.fit()
print(model5_fit.summary())

model6     = sm.OLS(x,y)
model6_fit = model6.fit()
print(model6_fit.summary())

# -------------------------------------------
# Q13
np.random.seed(1)

# (a)
x = np.random.normal(0, 1, size=(100,1))

# (b)
eps = np.random.normal(0, 0.5, size=(100,1)) 

# (c)
y = -1 + 0.5*x + eps
len(y) # length = 100
# b0 = -1, b1 = 0.5

# (d)
plt.scatter(x,y) # positively correlated
plt.xlabel("x")
plt.ylabel("y")

# (e)
x = sm.add_constant(x) # add intercept

model7     = sm.OLS(y,x)
model7_fit = model7.fit()
print(model7_fit.summary())
# intercept and coefficient clost to b0 and b1 in part (c)

# (f)
a     = model7_fit.params[0]
b     = model7_fit.params[1]
yfit  = [a + b*xi for xi in x[:,1]]
ytrue = [-1. + 0.5*xi for xi in x[:,1]]

plt.scatter(x[:,1],y)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x[:,1], yfit, 'blue', label='model fit')
plt.plot(x[:,1], ytrue, 'red', label='population')
plt.legend(loc='upper left')

# (g)
x2 = np.power(x,2)[:,1]
x2 = x2.reshape(100,1)
x  = np.append(x, x2, 1)

model8     = sm.OLS(y,x)
model8_fit = model8.fit()
print(model8.fit().summary())

table = anova_lm(model7.fit(), model8.fit())
print(table)
# No evidence on polynomial being better. 

# (h)-(j) skip

# -------------------------------------------
# Q14
# (a)
np.random.seed(1)
x1 = np.random.uniform(size=(100,))
x2 = 0.5*x1 + np.random.normal(size=(100,))/10.
y  = 2 + 2*x1 + 0.3*x2 + np.random.normal(size=(100,))

# (b)
np.corrcoef(x1, x2)
plt.scatter(x1, x2)
plt.xlabel("x1")
plt.ylabel("x2")

# (c)
x1 = x1.reshape(100,1)
x2 = x2.reshape(100,1)
x  = np.append(x1, x2, 1)
x  = sm.add_constant(x)

model9 = sm.OLS(y,x) 
print(model9.fit().summary())
# reject null for b1, fail to reject null for b2

# (d)
x       = sm.add_constant(x1)
model10 = sm.OLS(y,x) 
print(model10.fit().summary())
# reject

# (e)
x       = sm.add_constant(x2)
model11 = sm.OLS(y,x) 
print(model11.fit().summary())
# reject

# (f): No, because we need both x1 and x2 in the same regression. 

# (g)
x1 = np.append(x1, [[0.1]], 0)
x2 = np.append(x2, [[0.8]], 0)
y  = np.append(y, 6.0)

# regress on x1 and x2
x       = np.append(x1, x2, 1)
x       = sm.add_constant(x)
model12 = sm.OLS(y,x) 
print(model12.fit().summary())
# fail to reject null for b1, reject null for b2

model12_fit            = model12.fit()
model12_norm_residuals = model12_fit.get_influence().resid_studentized_internal
model12_leverage       = model12_fit.get_influence().hat_matrix_diag

plot1 = plt.figure(1)
plt.scatter(model12_leverage, model12_norm_residuals)
sns.regplot(model12_leverage, model12_norm_residuals, scatter=False, ci=False, lowess=True, line_kws={'color':'red', 'lw':1})
plot1.axes[0].set_title('Residuals vs Leverage')
plot1.axes[0].set_xlabel('Leverage')
plot1.axes[0].set_ylabel('Standardized residuals')

model12_leverage.argmax()
# new point high leverage for x1 and x2

# regress on x1
x       = sm.add_constant(x1)
model13 = sm.OLS(y,x) 
print(model13.fit().summary())
# reject

model13_fit            = model13.fit()
model13_norm_residuals = model13_fit.get_influence().resid_studentized_internal
model13_leverage       = model13_fit.get_influence().hat_matrix_diag

plot2 = plt.figure(2)
plt.scatter(model13_leverage, model13_norm_residuals)
sns.regplot(model13_leverage, model13_norm_residuals, scatter=False, ci=False, lowess=True, line_kws={'color':'red', 'lw':1})
plot2.axes[0].set_title('Residuals vs Leverage')
plot2.axes[0].set_xlabel('Leverage')
plot2.axes[0].set_ylabel('Standardized residuals')

model13_norm_residuals.argmax()
# new point outlier for x1

# regress on x2
x       = sm.add_constant(x2)
model14 = sm.OLS(y,x) 
print(model14.fit().summary())
# reject

model14_fit            = model14.fit()
model14_norm_residuals = model14_fit.get_influence().resid_studentized_internal
model14_leverage       = model14_fit.get_influence().hat_matrix_diag

plot3 = plt.figure(3)
plt.scatter(model14_leverage, model14_norm_residuals)
sns.regplot(model14_leverage, model14_norm_residuals, scatter=False, ci=False, lowess=True, line_kws={'color':'red', 'lw':1})
plot3.axes[0].set_title('Residuals vs Leverage')
plot3.axes[0].set_xlabel('Leverage')
plot3.axes[0].set_ylabel('Standardized residuals')

model14_leverage.argmax()
# new point high leverage for x2

# ----------------------------------------------------------------------------
# Q15
Boston = pd.read_csv('C:\\Users\\Carol\\Desktop\\Boston.csv') 

# (a)
results = []
y       = Boston.loc[:,'crim']
for i in range(2, 15, 1):
    name  = Boston.columns[i] 
    x     = Boston.loc[:, name]
    x     = sm.add_constant(x)
    model = sm.OLS(y,x)   
    results.append({name: model.fit().pvalues[1]})  

results
# chas is the only insignificant predictor.

# (b)
x         = Boston.iloc[:, 2:15]
x         = sm.add_constant(x)
y         = Boston.loc[:,'crim']
model_all = sm.OLS(y,x)   
print(model_all.fit().summary())
# reject for: zn, nox, dis, rad, black, istat, medv

# (c)
uni_coef = np.zeros(13)
y        = Boston.loc[:,'crim']
for i in range(2, 15, 1):
    name          = Boston.columns[i] 
    x             = Boston.loc[:, name]
    x             = sm.add_constant(x)
    model         = sm.OLS(y,x)   
    uni_coef[i-2] = model.fit().params[1]

mul_coef = model_all.fit().params[1:14]

plt.scatter(uni_coef, mul_coef)
plt.xlabel('uni coef')
plt.ylabel('mul coef')

uni_coef-mul_coef
# Coefficient estimates for nox are way off. 

# (d)
def poly(x, p):
    x = np.array(x)
    X = np.transpose(np.vstack((x**k for k in range(p+1))))
    return np.linalg.qr(X)[0][:,1:]

results = []
y       = Boston.loc[:,'crim']
for i in range(2, 15, 1):
    name  = Boston.columns[i] 
    temp  = Boston.loc[:, name]
    x1    = poly(temp, 3)[:, 0]
    x2    = poly(temp, 3)[:, 1]
    x3    = poly(temp, 3)[:, 2]
    x     = np.array([x1, x2, x3]).T
    x     = sm.add_constant(x)
    model = sm.OLS(y,x)   
    results.append(model.fit().pvalues[2:4])  

display            = pd.DataFrame(results, index=Boston.columns[2:15])
display['linear?'] = np.where((display.x2 < 0.1) | (display.x3 < 0.1), 'N', 'Y')
print(display)
# No evidence of non-linear association for chas and Black.