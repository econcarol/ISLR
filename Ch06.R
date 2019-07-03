# Ch 6 Applied
# -------------------------------------------
# Q8
rm(list=ls())

# (a)
set.seed(1)
X = rnorm(100)
e = rnorm(100)

# (b)
b = c(0.5, 1.2, 0.8, 3.6)
Y = b[1] + b[2]*X + b[3]*X^2 + b[4]*X^3 + e
plot(X,Y)

# (c)
df = data.frame(X,Y)

library (leaps) 
regfit.full = regsubsets(Y~poly(X,10,raw=T), data=df, nvmax=10) 
reg.summary = summary(regfit.full)

n1 = which.max(reg.summary$adjr2) #4
n2 = which.min(reg.summary$cp) #4
n3 = which.min(reg.summary$bic) #3

dev.new()
par(mfrow=c(3,1))
plot(reg.summary$adjr2, xlab="# of variables", ylab="adjusted R2", type="l")
points(n1, reg.summary$adjr2[n1], col="red", cex=2, pch=20) 
plot(reg.summary$cp, xlab="# of variables", ylab="Cp", type="l")
points(n2, reg.summary$cp[n2], col="red", cex=2, pch=20)
plot(reg.summary$bic, xlab="# of variables", ylab="BIC", type="l")
points(n3, reg.summary$bic[n3], col="red", cex=2, pch=20)

coef(regfit.full, n1)
coef(regfit.full, n2)
coef(regfit.full, n3)

# (d)
# forward
regfit.fwd   = regsubsets(Y~poly(X,10,raw=T), data=df, nvmax=10, method="forward") 
regf.summary = summary(regfit.fwd) 

n1 = which.max(regf.summary$adjr2) #4
n2 = which.min(regf.summary$cp) #4
n3 = which.min(regf.summary$bic) #3

dev.new()
par(mfrow=c(3,1))
plot(regf.summary$adjr2, xlab="# of variables", ylab="adjusted R2", type="l")
points(n1, regf.summary$adjr2[n1], col="red", cex=2, pch=20) 
plot(regf.summary$cp, xlab="# of variables", ylab="Cp", type="l")
points(n2, regf.summary$cp[n2], col="red", cex=2, pch=20)
plot(regf.summary$bic, xlab="# of variables", ylab="BIC", type="l")
points(n3, regf.summary$bic[n3], col="red", cex=2, pch=20)

coef(regfit.fwd, n1)
coef(regfit.fwd, n2)
coef(regfit.fwd, n3)

# backward
regfit.bwd   = regsubsets(Y~poly(X,10,raw=T), data=df, nvmax=10, method="backward")
regb.summary = summary(regfit.bwd) 

n1 = which.max(regb.summary$adjr2) #7
n2 = which.min(regb.summary$cp) #3
n3 = which.min(regb.summary$bic) #3

dev.new()
par(mfrow=c(3,1))
plot(regb.summary$adjr2, xlab="# of variables", ylab="adjusted R2", type="l")
points(n1, regb.summary$adjr2[n1], col="red", cex=2, pch=20) 
plot(regb.summary$cp, xlab="# of variables", ylab="Cp", type="l")
points(n2, regb.summary$cp[n2], col="red", cex=2, pch=20)
plot(regb.summary$bic, xlab="# of variables", ylab="BIC", type="l")
points(n3, regb.summary$bic[n3], col="red", cex=2, pch=20)

coef(regfit.bwd, n1)
coef(regfit.bwd, n2)
coef(regfit.bwd, n3)

# Best subset and forward agree, but backward differs.

# (e)
library(glmnet)
x = model.matrix(Y~poly(X,10,raw=T), df)[,-1] 
y = df$Y

set.seed(1) 
cv.out     = cv.glmnet(x, y, alpha=1)
plot(cv.out)
bestlam    = cv.out$lambda.min
lasso.coef = predict(cv.out, type="coefficients", s=bestlam)[1:11,]  
lasso.coef

# LASSO differs from the true model.

# (f)
Y  = 0.5 + 4.6*X^7 + e
df = data.frame(X,Y)

# best subset
regfit.full = regsubsets(Y~poly(X,10,raw=T), data=df, nvmax=10) 
reg.summary = summary(regfit.full)

n1 = which.max(reg.summary$adjr2) #4
n2 = which.min(reg.summary$cp) #2
n3 = which.min(reg.summary$bic) #1

dev.new()
par(mfrow=c(3,1))
plot(reg.summary$adjr2, xlab="# of variables", ylab="adjusted R2", type="l")
points(n1, reg.summary$adjr2[n1], col="red", cex=2, pch=20) 
plot(reg.summary$cp, xlab="# of variables", ylab="Cp", type="l")
points(n2, reg.summary$cp[n2], col="red", cex=2, pch=20)
plot(reg.summary$bic, xlab="# of variables", ylab="BIC", type="l")
points(n3, reg.summary$bic[n3], col="red", cex=2, pch=20)

coef(regfit.full, n1)
coef(regfit.full, n2)
coef(regfit.full, n3)

# LASSO
x = model.matrix(Y~poly(X,10,raw=T), df)[,-1] 
y = df$Y

set.seed(1) 
cv.out     = cv.glmnet(x, y, alpha=1)
plot(cv.out)
bestlam    = cv.out$lambda.min
lasso.coef = predict(cv.out, type="coefficients", s=bestlam)[1:11,] 
lasso.coef

# Best subset BIC and LASSO select the true one.

# -------------------------------------------
# Q9
rm(list=ls())

College = read.csv("C:\\Users\\Carol\\Desktop\\College.csv", header=T)
rownames(College) = College[,1] 
College = College[,-1]

# (a)
set.seed(1)
train    = sample(1:nrow(College), nrow(College)/2)
test     = (-train)
df.train = College[train,]
df.test  = College[test,]

# (b)
x.train = model.matrix(Apps~., df.train)[,-1]
y.train = df.train$Apps
x.test  = model.matrix(Apps~., df.test)[,-1]
y.test  = df.test$Apps

lm.mod  = glmnet(x.train, y.train, alpha=0, lambda=0)
lm.pred = predict(lm.mod, s=0, newx=x.test, exact=T) 
lm.err  = mean((lm.pred-y.test)^2) #1105066

# (c)
set.seed(1) 
cv.out  = cv.glmnet(x.train, y.train, alpha=0) 
plot(cv.out) 
bestlam = cv.out$lambda.min 

ridge.pred = predict(cv.out, s=bestlam, newx=x.test) 
ridge.err  = mean((ridge.pred-y.test)^2) #1037616

# (d)
set.seed(1) 
cv.out  = cv.glmnet(x.train, y.train, alpha=1) 
plot(cv.out) 
bestlam = cv.out$lambda.min 

lasso.pred = predict(cv.out, s=bestlam, newx=x.test) 
lasso.err  = mean((lasso.pred-y.test)^2) #1030941
lasso.coef = predict(cv.out, type="coefficients", s=bestlam)[1:18,] 
length(lasso.coef[lasso.coef!=0]) #16

# (e)
library(pls)

set.seed(1) 
pcr.fit = pcr(Apps~., data=df.train, scale=TRUE, validation="CV")
summary(pcr.fit) #16 comps
validationplot(pcr.fit, val.type="MSEP")
pcr.pred = predict(pcr.fit, x.test, ncomp=16)
pcr.err  = mean((pcr.pred-y.test)^2) #1166897

# (f)
set.seed(1) 
pls.fit = plsr(Apps~., data=df.train, scale=TRUE, validation="CV")
summary(pls.fit) #10 comps
validationplot(pls.fit, val.type="MSEP")
pls.pred = predict(pls.fit, x.test, ncomp=10)
pls.err  = mean((pls.pred-y.test)^2) #1134531

# (g)
err = c(lm.err, ridge.err, lasso.err, pcr.err, pls.err)
ind = c("lm", "ridge", "lasso", "pcr", "pls")
plot(err, xaxt="n", ylim=c(0, max(err)+1))
axis(1, at=1:length(ind), labels=ind)
# Not much difference; LASSO performs the best and PCR the worst.

# -------------------------------------------
# Q10
rm(list=ls())

# (a)
set.seed(1)
n = 1000
p = 20

e    = rnorm(n)
X    = replicate(p, rnorm(n))
b    = replicate(p, 0)
z    = sample(p,5, replace=FALSE)
b[z] = c(2, 0.5, 12, 1.4, 0.06)
Y    = X %*% b + e

# (b)
train    = sample(n, 100, replace=FALSE)
test     = (-train)
x.train  = X[train,]
x.test   = X[test,]
y.train  = Y[train,]
y.test   = Y[test,]
df.train = data.frame(y=y.train, x.train)
df.test  = data.frame(y=y.test, x.test)

# (c)
predict.regsubsets = function(object, newdata, id, ...){
  form  = as.formula(object$call[[2]]) 
  mat   = model.matrix(form, newdata) 
  coefi = coef(object, id=id) 
  xvars = names(coefi) 
  mat[,xvars] %*% coefi 
}

bsub  = regsubsets(y~., data=df.train, nvmax=p)
error = replicate(p,0)
for (i in 1:p){
  bsub.pred = predict.regsubsets(bsub, df.train, i)
  error[i]  = mean((y.train-bsub.pred)^2) 
}
plot(error, xlab='# of features', ylab='training MSE', type='l')
points(which.min(error), min(error), col="red", cex=2, pch=20)

# (d)
bsub  = regsubsets(y~., data=df.train, nvmax=p)
error = replicate(p,0)
for (i in 1:p){
  bsub.pred = predict.regsubsets(bsub, df.test, i)
  error[i]  = mean((y.test-bsub.pred)^2) 
}
plot(error, xlab='# of features', ylab='test MSE', type='l')
points(which.min(error), min(error), col="red", cex=2, pch=20)

# (e)
which.min(error) #4

# (f)
coef(bsub, which.min(error))
est   = coef[-1]
index = sort(z)
beta  = b[index]
mod   = cbind(index, beta)
print(mod)
print(est)
# Estimates are close to the true model. 

# (g)
r    = 20
bj   = matrix(b, p, r, dimnames=list(names(df.train[-1]), NULL))
bjr  = matrix(0, p, r, dimnames=list(names(df.train[-1]), NULL))
bsub = regsubsets(y~., data=df.train, nvmax=r)
for (i in 1:r){
  est = coef(bsub,i)[-1]
  bjr[names(est),i]  = est
}

val = sqrt(colSums((bj-bjr)^2))
plot(val, xlab='# of features', type='l')
points(which.min(val), min(val), col="red", cex=2, pch=20)
# Plot very similar to test MSE plot in (d).

# -------------------------------------------
# Q11
rm(list=ls())

library(MASS)
data(Boston)

set.seed(1)
train    = sample(c(TRUE, FALSE), nrow(Boston), rep=TRUE)
test     = (!train)
x.train  = model.matrix(crim~., data=Boston[train,])[,-1]
y.train  = Boston$crim[train]
df.train = Boston[train,]
x.test   = model.matrix(crim~., data=Boston[test,])[,-1]
y.test   = Boston$crim[test]
df.test  = Boston[test,]

# (a)
# best subset
predict.regsubsets = function(object, newdata, id, ...){
  form  = as.formula(object$call[[2]]) 
  mat   = model.matrix(form, newdata) 
  coefi = coef(object, id=id) 
  xvars = names(coefi) 
  mat[,xvars] %*% coefi 
}
# (i) cross-validation using training data to pick the best model
k = 10 
set.seed(1) 
folds  = sample(1:k, nrow(df.train), replace=TRUE) 
cv.err = matrix(NA, k, 13, dimnames=list(NULL, paste(1:13)))

for (j in 1:k){
  best.fit = regsubsets(crim~., data=df.train[folds!=j,], nvmax=13)
  for(i in 1:13){ 
    pred        = predict(best.fit, df.train[folds==j,], id=i)
    cv.err[j,i] = mean((df.train$crim[folds==j]-pred)^2)
  }
}

mean.cv.err = apply(cv.err, 2, mean) 
plot(mean.cv.err, xlab='# of features', ylab='MSE', type='b')
points(which.min(mean.cv.err), min(mean.cv.err), col="red", cex=2, pch=20) #2

# (ii) use the best model to get test error
bsub.fit  = regsubsets(crim~., data=df.train, nvmax=13)
bsub.coef = coef(bsub.fit, 2)
bsub.pred = predict.regsubsets(bsub.fit, df.test, 2)
bsub.err  = mean((y.test-bsub.pred)^2) #55.89

# ridge
set.seed(1) 
cv.out  = cv.glmnet(x.train, y.train, alpha=0) 
plot(cv.out) 
bestlam = cv.out$lambda.min 

ridge.pred = predict(cv.out, s=bestlam, newx=x.test) 
ridge.err  = mean((ridge.pred-y.test)^2) #58.27
ridge.coef = predict(cv.out, type="coefficients", s=bestlam)[1:14,]

# LASSO
set.seed(1) 
cv.out  = cv.glmnet(x.train, y.train, alpha=1) 
plot(cv.out) 
bestlam = cv.out$lambda.min 

lasso.pred = predict(cv.out, s=bestlam, newx=x.test) 
lasso.err  = mean((lasso.pred-y.test)^2) #56.27
lasso.coef = predict(cv.out, type="coefficients", s=bestlam)[1:14,] 
length(lasso.coef[lasso.coef!=0]) #7

# PCR
set.seed(1) 
pcr.fit = pcr(crim~., data=Boston, subset=train, scale=TRUE, validation="CV")
summary(pcr.fit) #13 comps
validationplot(pcr.fit, val.type="MSEP")
pcr.pred = predict(pcr.fit, x.test, ncomp=13)
pcr.err  = mean((pcr.pred-y.test)^2) #59.37

# (b)
err = c(bsub.err, ridge.err, lasso.err, pcr.err)
ind = c("best sub", "ridge", "lasso", "pcr")
plot(err, xaxt="n", ylab='test error')
axis(1, at=1:length(ind), labels=ind)
# Best subset performs the best and PCR the worst.

# (c) No, because some features hardly improve our prediction accuracy.  
