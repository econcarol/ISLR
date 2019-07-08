# Ch 7 Applied
# -------------------------------------------
# Q6
rm(list=ls())

library(ISLR)
fix(Wage)

# (a)
# CV
library(boot)
set.seed(1)
cv.error1 = rep(0,10)
for (i in 1:10){
  glm.fit      = glm(wage ~ poly(age,i), data=Wage) 
  cv.error1[i] = cv.glm(Wage, glm.fit, K=10)$delta[1] # 10-fold cv
}
minind = which.min(cv.error1) 
plot(cv.error1, xlab="# of polynomial degrees", ylab="10-fold CV MSE", type="b")
points(minind, cv.error1[minind], col="red", cex=2, pch=20)
# 4 is chosen by 10-fold CV.

# anova
fit.1  = lm(wage~age, data=Wage) 
fit.2  = lm(wage~poly(age,2), data=Wage)
fit.3  = lm(wage~poly(age,3), data=Wage) 
fit.4  = lm(wage~poly(age,4), data=Wage) 
fit.5  = lm(wage~poly(age,5), data=Wage) 
fit.6  = lm(wage~poly(age,6), data=Wage) 
fit.7  = lm(wage~poly(age,7), data=Wage) 
fit.8  = lm(wage~poly(age,8), data=Wage) 
fit.9  = lm(wage~poly(age,9), data=Wage) 
fit.10 = lm(wage~poly(age,10), data=Wage) 
anova(fit.1, fit.2, fit.3, fit.4, fit.5, fit.6, fit.7, fit.8, fit.9, fit.10)
# 4 is chosen by anova. 

# plot
agelims  = range(Wage$age)
age.grid = seq(from=agelims[1], to=agelims[2]) 
preds    = predict(fit.4, newdata=list(age=age.grid), se=TRUE)
se.bands = cbind(preds$fit+2*preds$se.fit, preds$fit-2*preds$se.fit)
dev.new()
par(mfrow=c(1,1), mar=c(4.5,4.5,1,1), oma=c(0,0,4,0))
plot(Wage$age, Wage$wage, xlim=agelims, xlab='age', ylab='wage', cex=.5, col='darkgrey') 
title('Degree-4 Polynomial', outer=T) 
lines(age.grid, preds$fit, lwd=2, col='blue') 
matlines(age.grid, se.bands, lwd=1, col='blue', lty=3)

# (b)
set.seed(1)
cv.error2 = rep(0, 9)
for (i in 2:10) {
  Wage$age.cut   = cut(Wage$age,i)
  glm.fit        = glm(wage~age.cut, data=Wage)
  cv.error2[i-1] = cv.glm(Wage, glm.fit, K=10)$delta[1] # 10-fold cv  
}
minind = which.min(cv.error2) 
plot(cv.error2, xlab='# of age bins', ylab='10-fold CV MSE', type='b')
points(minind, cv.error2[minind], col='red', cex=2, pch=20)
# 7 is chosen by 10-fold CV.

cut.fit  = glm(wage~cut(age,minind), data=Wage)
preds    = predict(cut.fit, newdata=list(age=age.grid), se=TRUE)
se.bands = cbind(preds$fit+2*preds$se.fit, preds$fit-2*preds$se.fit)
dev.new()
par(mfrow=c(1,1), mar=c(4.5,4.5,1,1), oma=c(0,0,4,0))
plot(Wage$age, Wage$wage, xlim=agelims, xlab='age', ylab='wage', cex=.5, col='darkgrey') 
title('7-Age-Bin Step Function', outer=T)
lines(age.grid, preds$fit, lwd=2, col='blue')
matlines(age.grid, se.bands, lwd=1, col='blue', lty=3)

# -------------------------------------------
# Q7
rm(list=ls())

plot(Wage$maritl, Wage$wage)
# Wage is highest for the married. 
plot(Wage$jobclass, Wage$wage)
# Wage is higher for information job class.

library(gam)
gam.fit1 = gam(wage~bs(age,df=4)+maritl, data=Wage)
gam.fit2 = gam(wage~bs(age,df=4)+jobclass, data=Wage)
gam.fit3 = gam(wage~bs(age,df=4)+maritl+jobclass, data=Wage)
anova(gam.fit1, gam.fit2, gam.fit3)
# Both marital status and job class should be included. 

dev.new()
par(mfrow=c(1,3), mar=c(4,4,1,1), oma=c(0,0,4,0))
plot(gam.fit3, se=TRUE, col='blue')
# GAM plots confirmed our observations of the box plots. 

# -------------------------------------------
# Q8
rm(list=ls())
fix(Auto)

dev.new()
pairs(Auto)
# non-linear relationship w/ mpg: displacement, horsepower, weight

set.seed(1)
cv.errord = rep(0,10)
for (i in 1:10) {
  glm.fit      = glm(mpg~poly(displacement,i), data=Auto)
  cv.errord[i] = cv.glm(Auto, glm.fit, K=10)$delta[1]
}
cv.errorh = rep(0,10)
for (i in 1:10) {
  glm.fit      = glm(mpg~poly(horsepower,i), data=Auto)
  cv.errorh[i] = cv.glm(Auto, glm.fit, K=10)$delta[1]
}
cv.errorw = rep(0,10)
for (i in 1:10) {
  glm.fit      = glm(mpg~poly(weight,i), data=Auto)
  cv.errorw[i] = cv.glm(Auto, glm.fit, K=10)$delta[1]
}

dev.new()
par(mfrow=c(3,1))
plot(cv.errord, main="displacement", xlab='# of polynomial degrees', ylab='10-fold CV MSE', type='b')
points(which.min(cv.errord), cv.errord[which.min(cv.errord)], col='red', cex=2, pch=20)
plot(cv.errorh, main="horsepower", xlab='# of polynomial degrees', ylab='10-fold CV MSE', type='b')
points(which.min(cv.errorh), cv.errorh[which.min(cv.errorh)], col='red', cex=2, pch=20)
plot(cv.errorw, main="weight", xlab='# of polynomial degrees', ylab='10-fold CV MSE', type='b')
points(which.min(cv.errorw), cv.errorw[which.min(cv.errorw)], col='red', cex=2, pch=20)

gam.fit1 = gam(mpg~poly(horsepower,6), data=Auto)
gam.fit2 = gam(mpg~poly(horsepower,6)+poly(displacement,10), data=Auto)
gam.fit3 = gam(mpg~poly(horsepower,6)+poly(weight,2), data=Auto)
gam.fit4 = gam(mpg~poly(horsepower,6)+poly(displacement,10)+poly(weight,2), data=Auto)
anova(gam.fit1, gam.fit2, gam.fit3, gam.fit4)
# All 3 should be included. 

par(mfrow=c(1,3))
plot(gam.fit4, se=TRUE, col='blue')
# The relation btw mpg and weight might be more linear. 

# -------------------------------------------
# Q9
rm(list=ls())

library(MASS)
fix(Boston)

# (a)
# regression results
fit.3 = lm(nox~poly(dis,3), data=Boston)
summary(fit.3)

# plot
dislims  = range(Boston$dis)
dis.grid = seq(from=dislims[1], to=dislims[2]) 
preds    = predict(fit.3, newdata=list(dis=dis.grid), se=TRUE)
se.bands = cbind(preds$fit+2*preds$se.fit, preds$fit-2*preds$se.fit)
dev.new()
par(mfrow=c(1,1), mar=c(4.5,4.5,1,1), oma=c(0,0,4,0))
# data plot
plot(Boston$dis, Boston$nox, xlim=dislims, xlab='distance', ylab='nitrogen oxides concentration', cex=.5, col='darkgrey') 
title('Degree-3 Polynomial', outer=T) 
# polynomial fit plot
lines(dis.grid, preds$fit, lwd=2, col='blue') 
matlines(dis.grid, se.bands, lwd=1, col='blue', lty=3)

# (b)
rss = rep(0,10)
for (i in 1:10) {
  lm.fit = lm(nox~poly(dis,i), data=Boston)
  rss[i] = sum(lm.fit$residuals^2)
}
plot(rss, type='b')
points(which.min(rss), rss[which.min(rss)], col='red')
# 10 is chosen.

# (c)
set.seed(1)
cv.error = rep(0,10)
for (i in 1:10) {
  glm.fit     = glm(nox~poly(dis,i), data=Boston)
  cv.error[i] = cv.glm(Boston, glm.fit, K=10)$delta[1]
}
plot(cv.error, type='b') 
points(which.min(cv.error), cv.error[which.min(cv.error)], col='red')
# 4 is chosen

# (d)
library(splines) 
fit  = lm(nox~bs(dis, df=4),data=Boston) 
pred = predict(fit, newdata =list(dis=dis.grid), se=T) 
attr(bs(Boston$dis, df=4),"knots") # 1 knot at 50%

plot(Boston$dis, Boston$nox, col='gray') 
lines(dis.grid, pred$fit ,lwd=2) 
lines(dis.grid, pred$fit +2*pred$se, lty='dashed') 
lines(dis.grid, pred$fit -2*pred$se, lty='dashed')

# (e)
set.seed(1)
rss = rep(0,7)
for (i in 4:10) {
  fit      = lm(nox~bs(dis, df=i), data=Boston)
  rss[i-3] = sum(fit$residuals^2)
}
plot(4:10, rss, xlab='degree of freedom', ylab='RSS', type='b')
# RSS decreases in df.

# (f)
fit.4  = lm(nox~bs(dis, df=4), data=Boston)
fit.5  = lm(nox~bs(dis, df=5), data=Boston)
fit.6  = lm(nox~bs(dis, df=6), data=Boston)
fit.7  = lm(nox~bs(dis, df=7), data=Boston)
fit.8  = lm(nox~bs(dis, df=8), data=Boston)
fit.9  = lm(nox~bs(dis, df=9), data=Boston)
fit.10 = lm(nox~bs(dis, df=10), data=Boston)
anova(fit.4, fit.5, fit.6, fit.7, fit.8, fit.9, fit.10)
# 2, 5 and 7 are the better choices.

# -------------------------------------------
# Q10
rm(list=ls())
fix(College)

# (a)
set.seed(1) 
n     = nrow(College)
p     = ncol(College)-1
train = sample(n, 0.5*n)

library (leaps)
regfit.fwd   = regsubsets(Outstate~., data=College[train,], nvmax=p, method="forward") 
regf.summary = summary(regfit.fwd) 

n1 = which.max(regf.summary$adjr2) #13
n2 = which.min(regf.summary$cp) #13
n3 = which.min(regf.summary$bic) #6

dev.new()
par(mfrow=c(3,1))
plot(regf.summary$adjr2, xlab='# of variables', ylab='adjusted R2', type='l')
points(n1, regf.summary$adjr2[n1], col='red', cex=2, pch=20) 
plot(regf.summary$cp, xlab='# of variables', ylab='Cp', type='l')
points(n2, regf.summary$cp[n2], col='red', cex=2, pch=20)
plot(regf.summary$bic, xlab='# of variables', ylab='BIC', type='l')
points(n3, regf.summary$bic[n3], col='red', cex=2, pch=20)

coef(regfit.fwd, n1)
coef(regfit.fwd, n2)
coef(regfit.fwd, n3)
# PrivateYes, Room.Board, Terminal, perc.alumni, Expend, Grad.Rate

# (b)
dev.new()
par(mfrow=c(2,3))
plot(College$Room.Board, College$Outstate) # seems linear
plot(College$Terminal, College$Outstate) # non-linear
plot(College$perc.alumni, College$Outstate) # non-linear
plot(College$Expend, College$Outstate) # non-linear
plot(College$Grad.Rate, College$Outstate) # non-linear

gam.fit1 = gam(Outstate~Private+Room.Board+Terminal+perc.alumni+Expend+Grad.Rate, data=College[train,])
gam.fit2 = gam(Outstate~Private+Room.Board+s(Terminal,6)+s(perc.alumni,6)+s(Expend,6)+s(Grad.Rate,6), data=College[train,])
gam.fit3 = gam(Outstate~Private+s(Room.Board,6)+s(Terminal,6)+s(perc.alumni,6)+s(Expend,6)+s(Grad.Rate,6), data=College[train,])
anova(gam.fit1, gam.fit2, gam.fit3)
# 2 and 3 are better. 

# (c)
predict.regsubsets = function(object, newdata, id, ...){
  form  = as.formula(object$call[[2]]) 
  mat   = model.matrix(form, newdata) 
  coefi = coef(object, id=id) 
  xvars = names(coefi) 
  mat[,xvars] %*% coefi 
}
fwd.pred = predict.regsubsets(regfit.fwd, College[-train,], n3)
fwd.mse  = mean((College[-train,]$Outstate-fwd.pred)^2) 
fwd.mse #4357411

gam.pred2 = predict(gam.fit2, College[-train,])
gam.mse2  = mean((College[-train,]$Outstate - gam.pred2)^2)
gam.mse2 #3760822

gam.pred3 = predict(gam.fit3, College[-train,])
gam.mse3  = mean((College[-train,]$Outstate - gam.pred3)^2)
gam.mse3 #3795890

# (d)
dev.new()
par(mfrow=c(2,3))
plot(gam.fit2, se=TRUE, col='blue')
# Terminal, perc.alumni, Expend, and Grad.Rate are non-linear.

# -------------------------------------------
# Q11
rm(list=ls())

# (a)
set.seed(1)
n  = 1000
x1 = rnorm(n)
x2 = rnorm(n)
e  = rnorm(n)
beta0 = 0.5
beta1 = 2.6
beta2 = 5.3
y  = beta0 + beta1*x1 + beta2*x2 + e

# (b)
beta1 = 4

# (c)
a     = y - beta1*x1
beta2 = lm(a~x2)$coef[2]

# (d)
a     = y - beta2*x2
beta1 = lm(a~x1)$coef[2]

# (e)
all.beta0 = rep(0,1000)
all.beta1 = rep(0,1000)
all.beta2 = rep(0,1000)

for (i in 1:1000){
  a            = y - all.beta1[i]*x1
  all.beta2[i] = lm(a~x2)$coef[2]
  
  a              = y - all.beta2[i]*x2
  all.beta1[i+1] = lm(a~x1)$coef[2]
  all.beta0[i]   = lm(a~x1)$coef[1]
}

dev.new()
plot(1:1000, all.beta0, ylim=c(0,6), col='blue', xlab='nth iteration', ylab='beta estimates')
points(1:1000, all.beta1[2:1001], col='red')
points(1:1000, all.beta2, col='green')
legend(800, 4, legend=c('beta0', 'beta1', 'beta2'), col=c('blue', 'red', 'green'), pch=1, cex=0.8)


# (f)
fit.lm = lm(y ~ x1 + x2)
coef(fit.lm)

dev.new()
plot(1:1000, all.beta0, ylim=c(0,6), col='blue', xlab='nth iteration', ylab='beta estimates')
points(1:1000, all.beta1[2:1001], col='red')
points(1:1000, all.beta2, col='green')
legend(800, 4, legend=c('beta0', 'beta1', 'beta2'), col=c('blue', 'red', 'green'), pch=1, cex=0.8)

abline(h=coef(fit.lm)[1], lty='dashed', lwd=3, col='orange')
abline(h=coef(fit.lm)[2], lty='dashed', lwd=3, col='black')
abline(h=coef(fit.lm)[3], lty='dashed', lwd=3, col='darkgray')
legend(200, 4, legend=c('beta0.lm', 'beta1,lm', 'beta2.lm'), col=c('orange', 'black', 'darkgray'), lty='dashed', cex=0.8)

# (g) One iteration is good enough based on graph in (f).

# -------------------------------------------
# Q12
rm(list=ls())

set.seed(1)
n = 1000     # # of obs
p = 100      # # of x's
c = rnorm(1) # intercept
b = rnorm(p) # coefficients
e = rnorm(n) # error term
x = replicate(p, rnorm(n))
y = c + x%*%b + e

# multiple linear regression results
fit.lm  = lm(y~x)
bhat.lm = rep(0,p)
for (j in 1:p){bhat.lm[j] = coef(fit.lm)[j+1]}

# backfitting results
ite  = 100 # # of iterations
mse  = rep(0,ite)
dif  = rep(0,ite)
chat = matrix(0, ite, p)
bhat = matrix(0, ite, p)

for (i in 1:ite){
  for (j in 1:p){
    a = y - x[,-j]%*%bhat[i,-j]
    # need to update values for ith iteration and onward
    chat[i:ite,j] = lm(a~x[,j])$coef[1]
    bhat[i:ite,j] = lm(a~x[,j])$coef[2] 
  }
  mse[i] = mean((y - chat[i,j] - x%*%bhat[i,])^2) 
  dif[i] = mean((bhat.lm - bhat[i,])^2)
}

dev.new()
par(mfrow=c(2,1))
plot(1:ite, mse, xlab='nth iteration', ylab='backfitting MSE for y') 
plot(1:ite, dif, xlab='nth iteration', ylab='lm vs. backfitting MSE for betas') 
# 3 ieration is good enough based on both graphs. 
