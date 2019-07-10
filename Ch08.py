# ISLR Ch 8 by Carol Cui

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error

# to use Graphviz
import graphviz
import os
os.environ['PATH'] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin'

# ----------------------------------------------------------------------------
# Q7
Boston = pd.read_csv('C:\\Users\\Carol\\Desktop\\Boston.csv')
Boston.head()

x = Boston.iloc[:,1:-1]
y = Boston.iloc[:,-1]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.5, random_state=1)

k   = 0
mse = np.zeros((10,6))
for j in range(25, 595, 95):
    for i in range(0,10):
        rf_boston = RandomForestRegressor(n_estimators=j, max_features=i+1, random_state=2)
        rf_boston.fit(xtrain, ytrain)
        ypred     = rf_boston.predict(xtest)
        mse[i,k]  = mean_squared_error(ytest, ypred)
    k += 1

ntree_grid = np.arange(25, 595, 95)
color_opts = ['black','blue','brown','green','red','gray','orange','pink','violet','yellow'] 

fig, ax = plt.subplots(1, 1)
for k in range(0,10):
    ax.plot(ntree_grid, mse[k,:], color=color_opts[k], label='m=%i' %(k+1))
ax.set_xlabel('# of trees')
ax.set_ylabel('test MSE')
ax.set_xticks(ntree_grid)
ax.legend()
# lowest test MSE: feature=7, ntree=25

# ----------------------------------------------------------------------------
# Q8
Carseats = pd.read_csv('C:\\Users\\Carol\\Desktop\\Carseats.csv')
Carseats.head()

Carseats['Urban']     = Carseats.Urban.map({'No':0, 'Yes':1})
Carseats['US']        = Carseats.US.map({'No':0, 'Yes':1})
Carseats['ShelveLoc'] = pd.factorize(Carseats.ShelveLoc)[0]
Carseats              = Carseats.drop('Unnamed: 0', axis=1)

# (a)
x = Carseats.drop(['Sales'], axis=1)
y = Carseats['Sales']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.5, random_state=1)

# (b)
tree_carseats = DecisionTreeRegressor()
tree_carseats.fit(xtrain, ytrain)

ypred = tree_carseats.predict(xtest)
mean_squared_error(ytest, ypred)

dot_data = export_graphviz(tree_carseats, out_file=None, feature_names=xtrain.columns)
graphviz.Source(dot_data)

# (c)
depth = []
for i in range(1,11):
    cv_tree = DecisionTreeRegressor(max_depth=i)
    scores  = cross_val_score(estimator=cv_tree, X=xtrain, y=ytrain, cv=10) # 10-fold CV
    depth.append(scores.mean())
plt.plot(range(1,11), depth)

pruned_tree_carseats = DecisionTreeRegressor(max_depth=depth.index(max(depth))+1)
pruned_tree_carseats.fit(xtrain, ytrain)

ypred = pruned_tree_carseats.predict(xtest)
mean_squared_error(ytest, ypred)

dot_data = export_graphviz(pruned_tree_carseats, out_file=None, feature_names=xtrain.columns)
graphviz.Source(dot_data)

# (d)
bag_carseats = RandomForestRegressor(max_features=len(xtrain.columns), random_state=2)
bag_carseats.fit(xtrain, ytrain)

ypred = bag_carseats.predict(xtest)
mean_squared_error(ytest, ypred)

Importance = pd.DataFrame({'Importance':bag_carseats.feature_importances_*100}, index=xtrain.columns)
Importance.sort_values(by='Importance', axis=0, ascending=True).plot(kind='barh', color='r')
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
# most important: Price, ShelveLoc

# (e)
mse = np.zeros(10)
for i in range(0,10):
    rf_carseats = RandomForestRegressor(max_features=i+1, random_state=2)
    rf_carseats.fit(xtrain, ytrain)
    ypred  = rf_carseats.predict(xtest)
    mse[i] = mean_squared_error(ytest, ypred)
    
plt.plot(range(1,11), mse)
plt.scatter(mse.argmin()+1, min(mse), color='r')
plt.xlabel('# of predictors')   
plt.ylabel('test MSE')
# Optimal # of predictors is 8.

rf_carseats = RandomForestRegressor(max_features=mse.argmin()+1, random_state=2)
rf_carseats.fit(xtrain, ytrain)
Importance = pd.DataFrame({'Importance':rf_carseats.feature_importances_*100}, index=xtrain.columns)
Importance.sort_values(by='Importance', axis=0, ascending=True).plot(kind='barh', color='r')
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
# most important: Price, ShelveLoc

# ----------------------------------------------------------------------------
# Q9
OJ = pd.read_csv('C:\\Users\\Carol\\Desktop\\OJ.csv').drop('Unnamed: 0', axis=1)
OJ.head()

OJ['Purchase'] = pd.factorize(OJ.Purchase)[0]
OJ['Store7']   = OJ.Store7.map({'No':0, 'Yes':1})

# (a)
x = OJ.drop(['Purchase'], axis=1)
y = OJ['Purchase']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=800, random_state=1)

# (b) & (c)
# use max_depth option to make tree structure more manageable for answering the questions
tree_oj = DecisionTreeClassifier(max_depth=3)
tree_oj.fit(xtrain, ytrain)
print('training error rate: %.2f' %(1-tree_oj.score(xtrain, ytrain))) 
# The tree has 15% training error rate.

# parse the tree structure
n_nodes        = tree_oj.tree_.node_count
children_left  = tree_oj.tree_.children_left
children_right = tree_oj.tree_.children_right
feature        = tree_oj.tree_.feature
threshold      = tree_oj.tree_.threshold

node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves  = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    node_depth[node_id]   = parent_depth + 1

    # if we have a test node
    if (children_left[node_id] != children_right[node_id]):
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        is_leaves[node_id] = True

print("The binary tree structure has %s nodes and has the following tree structure:" %n_nodes)
for i in range(n_nodes):
    if is_leaves[i]:
        print("%snode=%i leaf node." %(node_depth[i] * "\t", i))
    else:
        print("%snode=%i test node: go to node %i if X[:, %i] <= %.2f else to node %i."
              %(node_depth[i] * "\t", i, children_left[i], feature[i], threshold[i], children_right[i],))

print('# of terminal nodes: %i' %sum(is_leaves))
# The tree has 8 terminal nodes.

# (d)
dot_data = export_graphviz(tree_oj, out_file=None, feature_names=xtrain.columns)
graphviz.Source(dot_data)

# (e)
ypred = tree_oj.predict(xtest)
cmat  = pd.DataFrame(confusion_matrix(ytest, ypred).T, index = ['No', 'Yes'], columns = ['No', 'Yes'])
print(cmat)
# test error rate = (31+18)/270 = 18%

# (f)
depth = np.zeros(10)
for i in range(1,11):
    cv_tree    = DecisionTreeClassifier(max_depth=i)
    scores     = cross_val_score(estimator=cv_tree, X=xtrain, y=ytrain, cv=10) # 10-fold CV
    depth[i-1] = scores.mean()
print('optimal depth: %i' %(depth.argmax()+1)) #4

# (g)
plt.plot(range(1,11), 1-depth)
plt.xlabel('tree depth')
plt.ylabel('cv error rate')

# (h) best depth: 4

# (i)
pruned_tree_oj = DecisionTreeClassifier(max_depth=depth.argmax()+1)
pruned_tree_oj.fit(xtrain, ytrain)

dot_data = export_graphviz(pruned_tree_oj, out_file=None, feature_names=xtrain.columns)
graphviz.Source(dot_data)

# (j)
unpruned_tree_oj = DecisionTreeClassifier()
unpruned_tree_oj.fit(xtrain, ytrain)

print('unpruned training error rate: %.2f' %(1-unpruned_tree_oj.score(xtrain, ytrain))) 
print('pruned training error rate: %.2f' %(1-pruned_tree_oj.score(xtrain, ytrain))) 
# Pruned tree has 15% training error rate, much higher than unpruned tree.

# (k)
ypred = unpruned_tree_oj.predict(xtest)
cmat1 = pd.DataFrame(confusion_matrix(ytest, ypred).T, index = ['No', 'Yes'], columns = ['No', 'Yes'])
print(cmat1)
# unpruned tree test error rate = (41+24)/270 = 24%

ypred = pruned_tree_oj.predict(xtest)
cmat2 = pd.DataFrame(confusion_matrix(ytest, ypred).T, index = ['No', 'Yes'], columns = ['No', 'Yes'])
print(cmat2)
# pruned tree test error rate = (33+18)/270 = 19%, lower than unpruned tree

# ----------------------------------------------------------------------------
# Q10
# (a)
Hitters = pd.read_csv('C:\\Users\\Carol\\Desktop\\Hitters.csv').drop('Unnamed: 0', axis=1)
Hitters = Hitters.dropna()
Hitters.head()

Hitters['League']    = Hitters.League.map({'A':0, 'N':1})
Hitters['NewLeague'] = Hitters.NewLeague.map({'A':0, 'N':1})
Hitters['Division']  = Hitters.Division.map({'E':0, 'W':1})
Hitters['logSal']    = np.log(Hitters.Salary)

# (b)
x = Hitters.drop(['Salary','logSal'], axis=1)
y = Hitters['logSal']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=200, random_state=1)

# (c)
lambda_grid = np.linspace(0.001,0.201,20)
mse         = np.zeros(20)
for i in range(0,20):
    boost_hitters = GradientBoostingRegressor(n_estimators=1000, learning_rate=lambda_grid[i], random_state=2)
    boost_hitters.fit(xtrain, ytrain)
    mse[i] = mean_squared_error(ytrain, boost_hitters.predict(xtrain))

plt.plot(lambda_grid, mse)
plt.xlabel('lambda')
plt.ylabel('training MSE') 
plt.xticks(lambda_grid)

# (d)
lambda_grid = np.linspace(0.001,0.201,20)
mse         = np.zeros(20)
for i in range(0,20):
    boost_hitters = GradientBoostingRegressor(n_estimators=1000, learning_rate=lambda_grid[i], random_state=2)
    boost_hitters.fit(xtrain, ytrain)
    mse[i] = mean_squared_error(ytest, boost_hitters.predict(xtest))

plt.plot(lambda_grid, mse)
plt.xlabel('lambda')
plt.ylabel('test MSE') 
plt.xticks(lambda_grid)

# (e)
# multiple linear regression from Ch3
lm_hitter = LinearRegression().fit(xtrain, ytrain)
lm_mse    = mean_squared_error(ytest, lm_hitter.predict(xtest)) 

# lasso from Ch6
lasso_hitter = LassoCV(cv=10, normalize=True).fit(xtrain, ytrain)
lasso_mse    = mean_squared_error(ytest, lasso_hitter.predict(xtest)) 

# plot
x_label = np.arange(3)
plt.bar(x_label, [max(mse), lm_mse, lasso_mse])
plt.xticks(x_label, ('max boost', 'lm', 'lasso'))
plt.ylabel('test MSE')
# Boosting performs the best.

# (f)
best_lambda = lambda_grid[mse.argmin()]
best_boost_hitters = GradientBoostingRegressor(n_estimators=1000, learning_rate=best_lambda, random_state=2)
best_boost_hitters.fit(xtrain, ytrain)

f_imp   = best_boost_hitters.feature_importances_*100
rel_imp = pd.Series(f_imp, index=xtrain.columns).sort_values(inplace=False)
rel_imp.T.plot(kind='barh', color='r')
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
# CAtBat is the most important predictor. 

# (g)
bag_hitters = RandomForestRegressor(max_features=len(xtrain.columns), random_state=2)
bag_hitters.fit(xtrain, ytrain)
bag_mse     = mean_squared_error(ytest, bag_hitters.predict(xtest))

# ----------------------------------------------------------------------------
# Q11
Caravan = pd.read_csv('C:\\Users\\Carol\\Desktop\\Caravan.csv').drop('Unnamed: 0', axis=1)
Caravan.head()

Caravan['Purchase'] = Caravan.Purchase.map({'No':0, 'Yes':1})

# (a)
x = Caravan.drop(['Purchase'], axis=1)
y = Caravan['Purchase']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=1000, random_state=1)

# (b)
boost_caravan = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01, random_state=2)
boost_caravan.fit(xtrain, ytrain)

f_imp   = boost_caravan.feature_importances_*100
rel_imp = pd.Series(f_imp, index=xtrain.columns).sort_values(inplace=False)
rel_imp.T.plot(kind='barh', color='r')
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
# MGODOV is the most important predictor.

# (c)
# boost
ypred_prob = boost_caravan.predict_proba(xtest)[:,1]
ypred      = np.where(ypred_prob>0.2, 1, 0)

cmat1 = pd.DataFrame(confusion_matrix(ytest, ypred).T, index = ['No', 'Yes'], columns = ['No', 'Yes'])
print(cmat1)
# % of ppl predicted to purchase actually purchase: 48/290 = 16.55%

# logit
lm_caravan = LogisticRegression().fit(xtrain, ytrain)
ypred_prob = lm_caravan.predict_proba(xtest)[:,1]
ypred      = np.where(ypred_prob>0.2, 1, 0)

cmat2 = pd.DataFrame(confusion_matrix(ytest, ypred).T, index = ['No', 'Yes'], columns = ['No', 'Yes'])
print(cmat2)
# % of ppl predicted to purchase actually purchase: 60/350 = 17.14%

# KNN
knn_caravan = KNeighborsClassifier(n_neighbors=5).fit(xtrain, ytrain)
ypred_prob  = knn_caravan.predict_proba(xtest)[:,1]
ypred       = np.where(ypred_prob>0.2, 1, 0)

cmat3 = pd.DataFrame(confusion_matrix(ytest, ypred).T, index = ['No', 'Yes'], columns = ['No', 'Yes'])
print(cmat3)
# % of ppl predicted to purchase actually purchase: 36/183 = 19.67%

# Boosting performs the best, while KNN performs the worst.

# ----------------------------------------------------------------------------
# Q12
College = pd.read_csv('C:\\Users\\Carol\\Desktop\\College.csv').drop('Unnamed: 0', axis=1)
College['Private'] = College.Private.map({'No':0, 'Yes':1})

x = College.drop(['Grad.Rate'], axis=1)
y = College['Grad.Rate']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.5, random_state=1)

# boost
boost_college = GradientBoostingRegressor(random_state=2)
boost_college.fit(xtrain, ytrain)
boost_mse = mean_squared_error(ytest, boost_college.predict(xtest))

# bag
bag_college = RandomForestRegressor(max_features=len(xtrain.columns), random_state=3)
bag_college.fit(xtrain, ytrain)
bag_mse = mean_squared_error(ytest, bag_college.predict(xtest)) 

# random forest
# Default max_features is p/3 in R, so use 6 here.
rf_college = RandomForestRegressor(max_features=6, random_state=4)
rf_college.fit(xtrain, ytrain)
rf_mse = mean_squared_error(ytest, rf_college.predict(xtest)) 

# linear
lm_college = LinearRegression().fit(xtrain, ytrain)
lm_mse     = mean_squared_error(ytest, lm_college.predict(xtest)) 

# plot
x_label = np.arange(4)
plt.bar(x_label, [boost_mse, bag_mse, rf_mse, lm_mse])
plt.xticks(x_label, ('boost', 'bag', 'rf', 'lm'))
plt.ylabel('test MSE')

# Linear regression performs the best.
