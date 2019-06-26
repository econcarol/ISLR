# ISLR Ch 5 by Carol Cui
%reset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# ----------------------------------------------------------------------------
# Q5
Default = pd.read_csv('C:\\Users\\Carol\\Desktop\\Default.csv')

# (a)
x01 = sm.add_constant(Default.iloc[:, 3:5]) 
y01 = np.where(Default['default']=='No', 0, 1) 

glm1 = sm.Logit(y01, x01)
print(glm1.fit().summary())

# (b)
# (i)
x = pd.DataFrame(Default.iloc[:, 3:5])
y = np.array(Default['default'])
np.random.seed(1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
# (ii)
glm2 = LogisticRegression()
# (iii)
glm2.pred = glm2.fit(x_train, y_train).predict(x_test)
# (iv)
error1 = 1-accuracy_score(y_test, glm2.pred) # 3.1%
print(error1)

# (c)
error2 = np.zeros(3)
for i in range(2, 5, 1):
    np.random.seed(i)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    
    glm2        = LogisticRegression()
    glm2.pred   = glm2.fit(x_train, y_train).predict(x_test)
    error2[i-2] = 1-accuracy_score(y_test, glm2.pred)
    
error2
# Testing error is btw 3.1% and 3.4% (small variance).

# (d)
Default['student01'] = np.where(Default['student'] == 'No', 0, 1)
x = pd.DataFrame(Default.iloc[:, 3:6])
y = np.array(Default['default'])

error3 = np.zeros(4)
for i in range(1, 5, 1):
    np.random.seed(i)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
    
    glm3        = LogisticRegression()
    glm3.pred   = glm3.fit(x_train, y_train).predict(x_test)
    error3[i-1] = 1-accuracy_score(y_test, glm3.pred)
    
error3
# Testing error is similar to w/o student.

# ----------------------------------------------------------------------------
# Q6
# (a)
x01  = sm.add_constant(Default.iloc[:, 3:5]) 
y01  = np.where(Default['default']=='No', 0, 1) 
glm1 = sm.Logit(y01, x01)
print(glm1.fit().bse)
# Standard error is 0.000005 for income and 0.000227 for balance.

# (b)
def coef(x,y):
  glm = sm.Logit(y, x)  
  return glm.fit().params
coef(x01,y01)

# (c)
def boot(data, n):
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    
    for i in range(0, n):
        df    = data.sample(frac=1, replace=True)
        x     = sm.add_constant(df.iloc[:, 3:5])
        y     = np.where(df['default']=='No', 0, 1) 
        x1[i] = coef(x,y)[1] 
        x2[i] = coef(x,y)[2]
        
    res1 = np.std(x1)
    res2 = np.std(x2)
    print('balance se: %.8f; income se: %.8f' %(res1, res2))
    
boot(Default, 50)

# (d)
# Bootstrap standard errors are close to glm estimates. 

# ----------------------------------------------------------------------------
# Q7
Weekly = pd.read_csv('C:\\Users\\Carol\\Desktop\\Weekly.csv')

# (a)
x01  = sm.add_constant(Weekly.iloc[:, 2:4]) 
y01  = np.where(Weekly['Direction']=='Up', 1, 0) 
glm1 = sm.Logit(y01, x01)
print(glm1.fit().summary())

# (b) & (c)
x       = pd.DataFrame(Weekly.iloc[:, 2:4])
y       = np.array(Weekly['Direction'])
x_train = x.iloc[1:,:]
y_train = y[1:]
x_test  = x.iloc[0,:].reshape(1, -1)

glm2      = LogisticRegression()
glm2.pred = glm2.fit(x_train, y_train).predict(x_test)
print('actual: [\'%s\']; predicted: %s' %(y[0], glm2.pred))
# incorrectly classified

# (d)
n          = len(Weekly)
error_made = np.zeros(n)

for i in range(0, n):
    # (i)-(iii)
    x_train   = x.drop([i])
    y_train   = np.delete(y,i)
    x_test    = x.iloc[i,:].reshape(1, -1)
    glm2      = LogisticRegression()
    glm2.pred = glm2.fit(x_train, y_train).predict(x_test)
    
    # (iv)
    if glm2.pred != y[i]:
        error_made[i] = 1

# (e)
np.mean(error_made) # 45%

# ----------------------------------------------------------------------------
# Q8
# (a)
np.random.seed(1)
x = np.random.randn(100)
y = x - 2*x**2 + np.random.randn(100)
# n = 100 obs, p = 2 variables
# y = x - 2x^2 + random error

# (b)
plt.scatter(x, y) # y is hump-shaped in x.

# (c)
df = pd.DataFrame({'x': x, 'y': y})
x  = df['x'].values.reshape(-1,1)
y  = df['y'].values.reshape(-1,1)

lm   = LinearRegression()
mse1 = np.zeros(4)
for i in range(1,5):
    poly      = PolynomialFeatures(degree=i)
    x_poly    = poly.fit_transform(x)
    loocv     = KFold(n_splits=100, random_state=1)
    lm_fit    = lm.fit(x_poly, y)
    scores    = cross_val_score(lm_fit, x_poly, y, scoring="neg_mean_squared_error", cv=loocv)
    mse1[i-1] = np.mean(np.abs(scores))

mse1

# (d)
lm   = LinearRegression()
mse2 = np.zeros(4)
for i in range(1,5):
    poly      = PolynomialFeatures(degree=i)
    x_poly    = poly.fit_transform(x)
    loocv     = KFold(n_splits=100, random_state=2)
    lm_fit    = lm.fit(x_poly, y)
    scores    = cross_val_score(lm_fit, x_poly, y, scoring="neg_mean_squared_error", cv=loocv)
    mse2[i-1] = np.mean(np.abs(scores))

mse2
# Yes, results are exactly the same.
# Because LOOCV predicts every observation using the rest (aka no randomness).

# (e)
np.argmin(mse1) # Model iv
# No, because the true model was generated using x and x^2.

# (f)
poly   = PolynomialFeatures(degree=4)
x_poly = poly.fit_transform(x)  
x_poly = sm.add_constant(x_poly)   
lm     = sm.OLS(y, x_poly)
print(lm.fit().summary())
# x, x^2, and x^4 are statistically significant, consistent with CV results.

# ----------------------------------------------------------------------------
# Q9
Boston = pd.read_csv('C:\\Users\\Carol\\Desktop\\Boston.csv')

# (a)
mu_hat = np.mean(Boston.medv) 
print(mu_hat) # 22.53

# (b)
mu_hat_se = np.std(Boston.medv)/np.sqrt(len(Boston)) 
print(mu_hat_se) # 0.41

# (c)
def boot(var, n):
    m = np.zeros(n)
    for i in range(0, n):
        v    = var.sample(frac=1, replace=True)
        m[i] = np.mean(v)
    res1 = np.mean(m)
    res2 = np.std(m)
    print('mu: %.2f; se: %.2f' %(res1, res2))
    return(res1, res2)

result = boot(Boston.medv, 50) # close to (b)

# (d)
print('lowerbd:%.2f' %(result[0] - 2*result[1]))
print('upperbd:%.2f' %(result[0] + 2*result[1])) 

from scipy import stats
stats.t.interval(0.95,               # confidence level
                 df = len(Boston)-1, # degrees of freedom
                 loc = mu_hat,       # sample mean
                 scale= mu_hat_se)   # sample std dev

# (e)
mu_med_hat = np.median(Boston.medv)
print(mu_med_hat) # 21.2

# (f)
def boot(var, n):
    m = np.zeros(n)
    for i in range(0, n):
        v     = var.sample(frac=1, replace=True)
        m[i]  = np.median(v)
    r = np.std(m) 
    print(r)

result = boot(Boston.medv, 50)

# (g)
mu_10_hat = Boston['medv'].quantile(q=0.1)
print(mu_10_hat) # 12.75

# (h)
def boot(var, n):
    m = np.zeros(n)
    for i in range(0, n):
        v     = var.sample(frac=1, replace=True)
        m[i]  = v.quantile(q=0.1)
    r = np.std(m) 
    print(r)

result = boot(Boston.medv, 50)