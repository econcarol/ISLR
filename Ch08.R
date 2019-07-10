# Ch 8 Applied
# -------------------------------------------
# Q7
rm(list=ls())

library(MASS)
fix(Boston)

set.seed(1) 
train       = sample(1:nrow(Boston), nrow(Boston)/2)
boston.test = Boston[-train,'medv']

library(randomForest)

k     = 0
error = matrix(0, 10, 6)
for (j in seq(25,500,95)){
  k = k+1
  for (i in 1:10){
    rf.boston  = randomForest(medv~., data=Boston, subset=train, mtry=i, ntree=j)
    yhat.rf    = predict(rf.boston, newdata=Boston[-train,]) 
    error[i,k] = mean((yhat.rf-boston.test)^2)
  }
}

pred.num   = seq(1, 10)
ntree.grid = seq(25, 500, 95)
color.opts = c('black','blue','brown','green','red','gray','orange','pink','violet','yellow') 

dev.new()
plot(ntree.grid, error[1,], xaxt='n', xlab='# of trees', ylab='test MSE', type='l', ylim=c(floor(min(error)),ceiling(max(error)))) 
for (i in 2:10){lines(ntree.grid, error[i,], col=color.opts[i])}
axis(1, at=ntree.grid, las=2)
legend('topright', legend=paste0('m = ',pred.num), col=color.opts, lty=1)
# lowest test MSE: mtry=5, ntree=215

# -------------------------------------------
# Q8
rm(list=ls())

library(ISLR)
fix(Carseats)

# (a)
set.seed(1) 
train    = sample(1:nrow(Carseats), nrow(Carseats)/2)
car.test = Carseats[-train,'Sales']

# (b)
library(tree)
tree.car = tree(Sales~., Carseats, subset=train)
summary(tree.car)

dev.new()
plot(tree.car) 
text(tree.car, pretty=0)

# test MSE
tree.pred = predict(tree.car, newdata=Carseats[-train,]) 
mean((tree.pred-car.test)^2) #4.92

# (c)
set.seed(2)
cv.car     = cv.tree(tree.car) 
cv.size    = cv.car$size[which.min(cv.car$dev)]
prune.car  = prune.tree(tree.car, best=cv.size) 
prune.pred = predict(prune.car, newdata=Carseats[-train,]) 
mean((prune.pred-car.test)^2) #4.92
# Pruning doesn't improve test MSE cuz it selects the most complex tree. 

# (d)
set.seed(3) 
bag.car  = randomForest(Sales~., data=Carseats, subset=train, mtry=10, importance=T) 
bag.pred = predict(bag.car, newdata=Carseats[-train,]) 
mean((bag.pred-car.test)^2) #2.63

importance(bag.car)
varImpPlot(bag.car)
# most important: Price, ShelveLoc

# (e)
set.seed(3) 
rf.mse = rep(0,10)
for (i in 1:10){
  rf.car    = randomForest(Sales~., data=Carseats, subset=train, mtry=i, importance=T)
  rf.pred   = predict(rf.car, newdata=Carseats[-train,]) 
  rf.mse[i] = mean((rf.pred-car.test)^2)
}
 
dev.new()
plot(rf.mse, xlab='# of predictors', ylab='test MSE', type='l') 
points(which.min(rf.mse), min(rf.mse), col='red', cex=2, pch=20)
# Optimal # of predictors is 8.

rf.car = randomForest(Sales~., data=Carseats, subset=train, mtry=which.min(rf.mse), importance=T)
importance(rf.car)
varImpPlot(rf.car)
# most important: Price, ShelveLoc

# -------------------------------------------
# Q9
rm(list=ls())

library(ISLR)
fix(OJ)

# (a)
set.seed(1) 
train   = sample(1:nrow(OJ), 800)
oj.test = OJ[-train,'Purchase']

# (b)
tree.oj = tree(Purchase~., OJ, subset=train)
summary(tree.oj)
# The tree has 15.88% training error rate and 9 terminal nodes.

# (c)
tree.oj
# last node: split criterion = LoyalCH>0.764572, 261 obs in the branch, dev=91.2,
#            overall prediction = CH, (95.79% Yes, 4.22% No)

# (d)
dev.new()
plot(tree.oj) 
text(tree.oj, pretty=0)

# (e)
tree.pred = predict(tree.oj, OJ[-train,], type='class')
table(tree.pred, oj.test)
# test error rate = (38+8)/270 = 17%

# (f)
set.seed(2) 
cv.oj = cv.tree(tree.oj, FUN=prune.misclass)
cv.oj

# (g)
plot(cv.oj$size, cv.oj$dev, type='b')

# (h) best size: 8 or 9

# (i)
prune.oj = prune.misclass(tree.oj, best=5) 
plot(prune.oj) 
text(prune.oj, pretty=0)

# (j)
summary(prune.oj)
# Pruned tree has 16.25% training error rate, higher than unpruned tree.

# (k)
prune.pred = predict(prune.oj, OJ[-train,], type='class')
table(prune.pred, oj.test)
# test error rate = (36+8)/270 = 16%, lower than unpruned tree

# -------------------------------------------
# Q10
rm(list=ls())

library(ISLR)
fix(Hitters)

# (a)
Hitters        = na.omit(Hitters)
Hitters$logSal = log(Hitters$Salary)

# (b)
set.seed(1) 
train    = sample(1:nrow(Hitters), 200)
hit.test = Hitters[-train,'logSal']

# (c)
library(gbm)

set.seed(1) 
j         = 0
train.mse = rep(0, 20)
for (i in seq(0.001,0.201,0.01)){
  j = j + 1
  boost.hit    = gbm(logSal~.-Salary, data=Hitters[train,], distribution='gaussian', n.trees=1000, shrinkage=i)
  boost.pred   = predict(boost.hit, newdata=Hitters[train,], n.trees=1000, shrinkage=i) 
  train.mse[j] = mean((boost.pred-Hitters[train,'logSal'])^2)
}

lambda.grid = seq(0.001,0.201,0.01)
plot(lambda.grid, train.mse, xaxt='n', xlab='lambda', ylab='train MSE', type='b') 
axis(1, at=lambda.grid, las=2)

# (d)
set.seed(1) 
j        = 0
test.mse = rep(0, 20)
for (i in seq(0.001,0.201,0.01)){
  j = j + 1
  boost.hit   = gbm(logSal~.-Salary, data=Hitters[train,], distribution='gaussian', n.trees=1000, shrinkage=i)
  boost.pred  = predict(boost.hit, newdata=Hitters[-train,], n.trees=1000, shrinkage=i) 
  test.mse[j] = mean((boost.pred-hit.test)^2)
}

lambda.grid = seq(0.001,0.201,0.01)
plot(lambda.grid, test.mse, xaxt='n', xlab='lambda', ylab='test MSE', type='b') 
axis(1, at=lambda.grid, las=2) # best 0.011

# (e)
# multiple linear regression from Ch3
lm.hit  = lm(logSal~.-Salary, data=Hitters[train,])
lm.pred = predict(lm.hit, Hitters[-train,])
lm.mse  = mean((lm.pred-hit.test)^2) #0.48

# lasso from Ch6
library(glmnet)
x.train = model.matrix(logSal~.-Salary, Hitters[train,])[,-1]
y.train = Hitters[train,'logSal']
x.test  = model.matrix(logSal~.-Salary, Hitters[-train,])[,-1] 

set.seed(1) 
cv.out     = cv.glmnet(x.train, y.train, alpha=1)
bestlam    = cv.out$lambda.min
lasso.pred = predict(cv.out, s=bestlam, newx=x.test) 
lasso.mse  = mean((lasso.pred-hit.test)^2) #0.47

barplot(c(max(test.mse), lm.mse, lasso.mse), xaxt='n', ylab='test MSE')
axis(1, at=1:3, labels=c('max boosting', 'lm', 'lasso'))
# Boosting performs the best.

# (f)
best.boost.hit = gbm(logSal~.-Salary, data=Hitters[train,], distribution='gaussian', n.trees=1000, shrinkage=0.011)
summary(best.boost.hit)
# CRuns is the most important predictor. 

# (g)
library(randomForest) 
set.seed(1) 
bag.hit  = randomForest(logSal~.-Salary, data=Hitters[train,], mtry=19) 
bag.pred = predict(bag.hit, newdata=Hitters[-train,]) 
bag.mse  = mean((bag.pred-hit.test)^2) #0.25

# -------------------------------------------
# Q11
rm(list=ls())

library(ISLR)
fix(Caravan)

# (a)
Caravan$Pur01 = ifelse(Caravan$Purchase=='Yes', 1, 0)

set.seed(1) 
train    = sample(1:nrow(Caravan), 1000)
can.test = Caravan[-train,'Pur01']

# (b)
boost.can = gbm(Pur01~.-Purchase, data=Caravan[train,], distribution='bernoulli', n.trees=1000, shrinkage=0.01)
summary(boost.can)
# PPERSAUT is the most important predictor. 

# (c)
boost.pred   = predict(boost.can, newdata=Caravan[-train,], distribution='bernoulli', n.trees=1000, shrinkage=0.01) 
boost.pred01 = rep(0, nrow(Caravan[-train,])) 
boost.pred01[boost.pred>.2] = 1
table(boost.pred01,can.test)
# % of ppl predicted to purchase actually purchase: 0%

# logit
glm.can  = glm(Pur01~.-Purchase, data=Caravan[train,], family=binomial)
glm.prob = predict(glm.can, Caravan[-train,], type='response')
glm.pred = rep(0, nrow(Caravan[-train,]))  
glm.pred[glm.prob>.2] = 1
table(glm.pred,can.test)
# % of ppl predicted to purchase actually purchase: 52/(319+52) = 14.02%

# KNN
library(class) 
x.train = model.matrix(Pur01~.-Purchase, Caravan[train,])[,-1]
y.train = Caravan[train,'Pur01']
x.test  = model.matrix(Pur01~.-Purchase, Caravan[-train,])[,-1]

set.seed(1) 
knn.pred = knn(x.train, x.test, y.train, k=5, prob=T)
knn.prob = attr(knn.pred, 'prob')
new.pred = knn.pred
# use 0.2 as the probability threshold for Purchase
for (i in 1:nrow(x.test)){
  if (knn.pred[i] == 0 & (1-knn.prob[i])>0.2){
    new.pred[i] = 1
  }
}
table(new.pred,can.test)
# % of ppl predicted to purchase actually purchase: 27/(137+27) = 16.46%
# Logit performs the best, while boosting performs the worst. 

# -------------------------------------------
# Q12
rm(list=ls())

library(ISLR)
fix(College)

set.seed(1)
train = sample(1:nrow(College), nrow(College)/2)

# boosting
library(gbm)
boost.fit  = gbm(Grad.Rate~., data=College[train,], distribution='gaussian', n.trees=1000)
boost.pred = predict(boost.fit, newdata=College[-train,], n.trees=1000) 
boost.mse  = mean((boost.pred-College[-train,'Grad.Rate'])^2) #194.58

# bagging
library(randomForest)
set.seed(2) 
bag.fit  = randomForest(Grad.Rate~., data=College[train,], mtry=17) 
bag.pred = predict(bag.fit, newdata=College[-train,]) 
bag.mse  = mean((bag.pred-College[-train,'Grad.Rate'])^2) #164.43

# random forest
set.seed(3) 
rf.fit  = randomForest(Grad.Rate~., data=College[train,])
rf.pred = predict(rf.fit, newdata=College[-train,]) 
rf.mse  = mean((rf.pred-College[-train,'Grad.Rate'])^2) #160.57

# linear
lm.fit  = lm(Grad.Rate~., data=College[train,])
lm.pred = predict(lm.fit, College[-train,])
lm.mse  = mean((lm.pred-College[-train,'Grad.Rate'])^2) #168.47

# plot
barplot(c(boost.mse, bag.mse, rf.mse, lm.mse), xaxt='n', ylab='test MSE')
axis(1, at=1:4, labels=c('boost', 'bag', 'rf', 'lm'))
# Random forest w/ default parameters performs the best.
