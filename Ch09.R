# Ch 9 Applied
# -------------------------------------------
# Q4
rm(list=ls())

set.seed(1)
n = 100
p = 2
y = c(rep(1,33), rep(2,67)) 
x = matrix(rnorm(n*p), ncol=p) 
x[1:33,]  = x[1:33,]  + 2 
x[34:60,] = x[34:60,] - 2 
df = data.frame(x=x, y=as.factor(y))
plot(x, col=y)

set.seed(2)
train    = sample(1:n, n/2)
df.train = df[train,]
df.test  = df[-train,]

# SVC
library(e1071)
svcfit = svm(y~., data=df.train, kernel='linear', cost=10, scale=F)
table(predict=svcfit$fitted, truth=df.train$y) # training error = 3/50 = 6%
plot(svcfit, df.train)

# SVM poly
svmpfit = svm(y~., data=df.train, kernel='polynomial', degree=3, cost=10, scale=F) 
table(predict=svmpfit$fitted, truth=df.train$y) # training error = 0/50 = 0%
plot(svmpfit, df.train)

# SVM radial
svmrfit = svm(y~., data=df.train, kernel='radial', gamma=1, cost=10, scale=F) 
table(predict=svmrfit$fitted, truth=df.train$y) # training error = 0/50 = 0%
plot(svmrfit, df.train)

# SVM radial and poly outperform SVC w.r.t. training error. 

# testing error
ypred  = predict(svcfit, df.test) 
table(predict=ypred, truth=df.test$y) # training error = 4/50 = 8%
plot(svcfit, df.test)

ypred  = predict(svmpfit, df.test) 
table(predict=ypred, truth=df.test$y) # training error = 6/50 = 12%
plot(svmpfit, df.test)

ypred  = predict(svmrfit, df.test) 
table(predict=ypred, truth=df.test$y) # training error = 4/50 = 8%
plot(svmrfit, df.test)

# SVC and SVM radial perform equally well on test data. 

# -------------------------------------------
# Q5
rm(list=ls())

# (a)
set.seed(1)
n = 500
p = 2

x1 = runif(n) - 0.5
x2 = runif(n) - 0.5
x  = cbind(x1,x2)
y  = 1*(x1^2-x2^2 > 0)

df = data.frame(x, y=as.factor(y))

# (b)
plot(x, col=c('red','blue')[y+1])

# (c)
logit1 = glm(y~., data=df, family=binomial)
summary(logit1)

# (d)
set.seed(2)
train    = sample(1:n, n/2)
df.train = df[train,]

contrasts(df$y)
logit1.prob = predict(logit1, df.train, type='response')
logit1.pred = rep(0, nrow(df.train)) 
logit1.pred[logit1.prob > .5] = 1

plot(df.train$x1, df.train$x2, col=c('red','blue')[logit1.pred+1])

# (e)
logit2 = glm(y ~ I(poly(x1,2,raw=T)) + I(poly(x2,2,raw=T)), data=df, family=binomial)
summary(logit2)

# (f)
logit2.prob = predict(logit2, df.train, type='response')
logit2.pred = rep(0, nrow(df.train)) 
logit2.pred[logit2.prob > .5] = 1

plot(df.train$x1, df.train$x2, col=c('red','blue')[logit2.pred+1])

# (g)
svc      = svm(y~., data=df, kernel='linear', cost=10, scale=F)
svc.pred = predict(svc, df.train)

plot(df.train$x1, df.train$x2, col=c('red','blue')[svc.pred])

# (h)
svm      = svm(y~., data=df, kernel='polynomial', degree=2, cost=10, scale=F) 
svm.pred = predict(svm, df.train) 

plot(df.train$x1, df.train$x2, col=c('red','blue')[svm.pred])

# (i)
dev.new()
par(mfrow=c(2,2))
plot(df.train$x1, df.train$x2, col=c('red','blue')[logit1.pred+1], main='linear logit')
plot(df.train$x1, df.train$x2, col=c('red','blue')[svc.pred], main='SVC')
plot(df.train$x1, df.train$x2, col=c('red','blue')[logit2.pred+1], main='poly-degree-2 logit')
plot(df.train$x1, df.train$x2, col=c('red','blue')[svm.pred], main='poly-degree-2 SVM')
# While SVC and linear logit are similar, SVM poly and non-linear logit are similar. 

# -------------------------------------------
# Q6
rm(list=ls())

# (a)
set.seed(1)
n = 100
p = 2

x  = matrix(rnorm(n*p), ncol=p) 
y  = c(rep(1,30), rep(2,70)) 
x[y==1,] = x[y==1,] + 2.2
df = data.frame(x=x, y=as.factor(y))

plot(x, pch =19, col=c('red','blue')[y])

# (b)
ctry = c(0.1,1,10,100,1000)

# CV
set.seed(2)
tune.out = tune(svm, y~., data=df, kernel='linear', ranges=list(cost=ctry), scale=F)
summary(tune.out)
# best cost: 0.1
# CV error can go up as cost rises. 

# training errors
err = rep(0,5)
for (i in 1:5){
  svc    = svm(y~., data=df, kernel='linear', cost=ctry[i], scale=F)
  err[i] = sum(svc$fitted != df$y)/n
}
err
# best cost: 1
# Training error never rises in cost. 

# (c)
set.seed(3)
ntest = 100
xtest = matrix(rnorm(ntest*p), ncol=p) 
ytest = c(rep(1,30), rep(2,70)) 
xtest[ytest==1,] = xtest[ytest==1,] + 2.2 
df.test = data.frame(x=xtest, y=as.factor(ytest))
plot(xtest, pch =19, col=c('red','blue')[ytest])

test.err = rep(0,5)
for (i in 1:5){
  svc         = svm(y~., data=df, kernel='linear', cost=ctry[i], scale=F)
  pred        = predict(svc, df.test) 
  test.err[i] = sum(pred != df.test$y)/n
}
test.err
# best cost: 0.1
# Test error selection agrees with CV selection, not training error selection. 

# (d) The claim at the end of Section 9.6.1 is true. 

# -------------------------------------------
# Q7
rm(list=ls())

library(ISLR)
fix(Auto)

# (a)
mpg01 = ifelse(Auto$mpg > median(Auto$mpg), 1, 0)
df    = data.frame(x=Auto[,2:7], y=as.factor(mpg01))

# (b)
ctry = c(0.01,0.1,1,10,100,1000)

set.seed(1)
tune.out1 = tune(svm, y~., data=df, kernel='linear', ranges=list(cost=ctry))
summary(tune.out1)
# best cost: 100

# (c)
polytry = c(2,3,4,5)

set.seed(1)
tune.out2 = tune(svm, y~., data=df, kernel='polynomial', ranges=list(cost=ctry, degree=polytry))
summary(tune.out2)
# best combo: cost=10, degree=3

gammatry = c(0.5,1,2,3)

set.seed(1)
tune.out3 = tune(svm, y~., data=df, kernel='radial', ranges=list(cost=ctry, gamma=gammatry))
summary(tune.out3)
# best combo: cost=100, gamma=2
# lowest best CV error

# (d)
svc = tune.out1$best.model 
table(predict=svc$fitted, truth=df$y) # 30 wrong
plot(svc, df, x.year~x.weight)

svmp = tune.out2$best.model  
table(predict=svmp$fitted, truth=df$y) # 26 wrong
plot(svmp, df, x.year~x.weight)

svmr = tune.out3$best.model
table(predict=svmr$fitted, truth=df$y) # 0 wrong
plot(svmr, df, x.year~x.weight)

# SVM radial fits the training data best. 

# -------------------------------------------
# Q8
rm(list=ls())

library(ISLR)
fix(OJ)

# (a)
set.seed(1)
train  = sample(1:nrow(OJ), 800)
xtrain = OJ[train,-1]
ytrain = OJ[train,'Purchase']
xtest  = OJ[-train,-1]
ytest  = OJ[-train,'Purchase']

df.train = data.frame(x=xtrain, y=as.factor(ytrain))
df.test  = data.frame(x=xtest, y=as.factor(ytest))

# (b)
svc = svm(y~., data=df.train, kernel='linear', cost=0.01) 
summary(svc)

# (c)
svc.train.err = sum(svc$fitted != df.train$y)/800
ypred         = predict(svc, df.test)
svc.test.err  = sum(ypred != df.test$y)/nrow(df.test)
print(paste0('training error: ', svc.train.err, ' | test error: ', svc.test.err))

# (d)
ctry = c(0.01, 0.1, 1, 10)

set.seed(2)
tune.out1 = tune(svm, y~., data=df.train, kernel='linear', ranges=list(cost=ctry))
summary(tune.out1)

# (e)
best.svc       = tune.out1$best.model
bsvc.train.err = sum(best.svc$fitted != df.train$y)/800
ypred          = predict(best.svc, df.test)
bsvc.test.err  = sum(ypred != df.test$y)/nrow(df.test)
print(paste0('training error: ', bsvc.train.err, ' | test error: ', bsvc.test.err))

# (f)
svmr = svm(y~., data=df.train, kernel='radial', cost=0.01) 
summary(svmr)

svmr.train.err = sum(svmr$fitted != df.train$y)/800
ypred          = predict(svmr, df.test)
svmr.test.err  = sum(ypred != df.test$y)/nrow(df.test)
print(paste0('training error: ', svmr.train.err, ' | test error: ', svmr.test.err))

set.seed(2)
tune.out2 = tune(svm, y~., data=df.train, kernel='radial', ranges=list(cost=ctry))
summary(tune.out2)

best.svmr       = tune.out2$best.model
bsvmr.train.err = sum(best.svmr$fitted != df.train$y)/800
ypred           = predict(best.svmr, df.test)
bsvmr.test.err  = sum(ypred != df.test$y)/nrow(df.test)
print(paste0('training error: ', bsvmr.train.err, ' | test error: ', bsvmr.test.err))

# (g)
svmp = svm(y~., data=df.train, kernel='polynomial', degree=2, cost=0.01) 
summary(svmp)

svmp.train.err = sum(svmp$fitted != df.train$y)/800
ypred          = predict(svmp, df.test)
svmp.test.err  = sum(ypred != df.test$y)/nrow(df.test)
print(paste0('training error: ', svmp.train.err, ' | test error: ', svmp.test.err))

set.seed(2)
tune.out3 = tune(svm, y~., data=df.train, kernel='polynomial', degree=2, ranges=list(cost=ctry))
summary(tune.out3)

best.svmp       = tune.out3$best.model
bsvmp.train.err = sum(best.svmp$fitted != df.train$y)/800
ypred           = predict(best.svmp, df.test)
bsvmp.test.err  = sum(ypred != df.test$y)/nrow(df.test)
print(paste0('training error: ', bsvmp.train.err, ' | test error: ', bsvmp.test.err))

# (h)
err      = c(svc.test.err, svmr.test.err, svmp.test.err, bsvc.test.err, bsvmr.test.err, bsvmp.test.err) 
err.name = c('linear', 'radial', 'poly', 'best linear', 'best radial', 'best poly')
barplot(err, names.arg=err.name, ylab='test error')
# SVC performs the best. 
