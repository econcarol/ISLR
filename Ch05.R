# Ch 5 Applied
# -------------------------------------------
# Q5
rm(list=ls())

library(ISLR)
data(Default)

# (a)
contrasts(Default$default)
glm.fit1 = glm(default ~ income + balance, data=Default, family=binomial) 
summary(glm.fit1)

# (b)
# (i)
set.seed(1) 
n     = dim(Default)[1]
train = sample(n, 0.6*n)
test  = Default[-train,] 

# (ii)
glm.fit2  = glm(default ~ income + balance, data=Default, family=binomial, subset=train)

# (iii)
glm.prob2 = predict(glm.fit2, test, type='response')
glm.pred2 = rep('No', dim(test)[1]) 
glm.pred2[glm.prob2 > .5] = 'Yes'

# (iv)
error1 = mean(glm.pred2 != test$default) # 2.8%

# (c)
error2 = rep(0,3)
for (i in 2:4){
  set.seed(i)
  train = sample(n, 0.6*n)
  test  = Default[-train,] 
  
  glm.fit2  = glm(default ~ income + balance, data=Default, family=binomial, subset=train)
  glm.prob2 = predict(glm.fit2, test, type='response')
  glm.pred2 = rep('No', dim(test)[1]) 
  glm.pred2[glm.prob2 > .5] = 'Yes'
  
  error2[i-1] = mean(glm.pred2 != test$default)
}
error2
# Testing error is btw 2.4% and 2.8% (small variance).

# (d)
error3 = rep(0,4)
for (i in 1:4){
  set.seed(i)
  train = sample(n, 0.6*n)
  test  = Default[-train,] 
  
  glm.fit3  = glm(default ~ income + balance + student, data=Default, family=binomial, subset=train)
  glm.prob3 = predict(glm.fit3, test, type='response')
  glm.pred3 = rep('No', dim(test)[1]) 
  glm.pred3[glm.prob3 > .5] = 'Yes'
  
  error3[i] = mean(glm.pred3 != test$default)
}
error3
# Testing error is similar to w/o student.

# -------------------------------------------
# Q6
# (a)
glm.fit1 = glm(default ~ income + balance, data=Default, family=binomial) 
summary(glm.fit1)
# Standard error is 0.000004985 for income and 0.0002274 for balance.

# (b)
boot.fn = function(data, index){
  return(coef(glm(default ~ income + balance, data=data, family=binomial, subset=index))) 
}
boot.fn(Default, 1:n)

# (c)
library(boot)
set.seed(1) 
boot(Default, boot.fn, 50)

# (d)
# Bootstrap standard error is 0.000004542 for income and 0.0002283 for balance, close to glm estimates. 

# -------------------------------------------
# Q7
rm(list=ls())

library(ISLR)
data(Weekly)

# (a)
contrasts(Weekly$Direction)
glm.fit1 = glm(Direction ~ Lag1 + Lag2, data=Weekly, family=binomial)
summary(glm.fit1)

# (b)
glm.fit2 = glm(Direction ~ Lag1 + Lag2, data=Weekly, family=binomial, subset=2:nrow(Weekly))
summary(glm.fit2)

# (c)
prob = predict(glm.fit2, Weekly[1,], type='response')
pred = ifelse(prob > 0.5, 'Up', 'Down')
print(paste('actual:', Weekly$Direction[1], 'predicted:', pred))
# incorrectly classified

# (d)
n          = nrow(Weekly)
error.made = rep(0,n)
for (i in 1:n){
  # (i)
  glm.fit2 = glm(Direction ~ Lag1 + Lag2, data=Weekly[-i,], family=binomial)
  # (ii)
  prob = predict(glm.fit2, Weekly[i,], type='response')
  # (iii)
  pred = ifelse(prob > 0.5, 'Up', 'Down')
  # (iv)
  error.made[i] = ifelse(pred != Weekly$Direction[i], 1, 0)
}

# (e)
mean(error.made) # 45%

# -------------------------------------------
# Q8
rm(list=ls())

# (a)
set.seed(1)
x = rnorm(100)
y = x - 2*x^2 + rnorm(100)
# n = 100 obs, p = 2 variables
# y = x - 2x^2 + random error

# (b)
plot(x, y) # y is hump-shaped in x.

# (c)
set.seed(1)
df        = data.frame(x,y)
cv.error1 = rep(0,4)
for (i in 1:4){
  glm.fit      = glm(y ~ poly(x,i), data=df) 
  cv.error1[i] = cv.glm(df, glm.fit)$delta[1]
}
cv.error1

# (d)
set.seed(2)
cv.error2 = rep(0,4)
for (i in 1:4){
  glm.fit      = glm(y ~ poly(x,i), data=df) 
  cv.error2[i] = cv.glm(df, glm.fit)$delta[1]
}
cv.error2
# Yes, results are exactly the same.
# Because LOOCV predicts every observation using the rest (aka no randomness).

# (e)
which.min(cv.error1) # Model ii
# Yes, because the true model was generated using x and x^2.

# (f)
lm.fit = lm(y ~ poly(x,4), data=df)
summary(lm.fit)
# Only x and x^2 are statistically significant, consistent with CV results.

# -------------------------------------------
# Q9
rm(list=ls())

library(MASS)
data(Boston)

# (a)
mu.hat = mean(Boston$medv) #22.53

# (b)
mu.hat.se = sd(Boston$medv)/sqrt(nrow(Boston)) #0.41

# (c)
set.seed(1)
boot.fn   = function(var, i) return(mean(var[i]))
boot.stat = boot(Boston$medv, boot.fn, 50) #0.39, close to (b)
boot.stat

# (d)
print(paste('lowerbd:', boot.stat$t0 - 2*sd(boot.stat$t)))
print(paste('upperbd:', boot.stat$t0 + 2*sd(boot.stat$t))) 
t.test(Boston$medv)

# (e)
mu.med.hat = median(Boston$medv) #21.2

# (f)
set.seed(1)
boot.fn   = function(var, i) return(median(var[i]))
boot(Boston$medv, boot.fn, 50) #0.39

# (g)
mu.0.1.hat = quantile(Boston$medv, 0.1) #12.8

# (h)
set.seed(1)
boot.fn   = function(var, i) return(quantile(var[i], 0.1))
boot(Boston$medv, boot.fn, 50) #0.50
