# Ch 3 Applied
# -------------------------------------------
# Q8
rm(list=ls())

# (a)
auto = read.csv("C:\\Users\\Carol\\Desktop\\Auto.csv", header=T, na.strings='?')
auto = na.omit(auto)
fix(auto)

model1 = lm(mpg~horsepower, data=auto)
summary(model1)

# (i): Yes.
# (ii): Strong as p-value is very close to 0.
# (iii): Negative.
# (iv)
predict(model1, data.frame(horsepower=98), interval="confidence") 
predict(model1, data.frame(horsepower=98), interval="prediction")

# (b)
plot(auto$horsepower, auto$mpg)
abline(model1, lwd=3, col="red")

# (c)
dev.new()
par(mfrow=c(2,2))
plot(model1)

# -------------------------------------------
# Q9
# (a)
dev.new()
pairs(auto)

# (b)
cor(auto[,-9])

# (c)
model2 = lm(mpg~.-name, data=auto)
summary(model2)

# (i): Yes.
# (ii): displacement, weight, year, origin
# (iii): Newer models have higher mpg.

# (d)
dev.new()
par(mfrow=c(2,2))
plot(model2)
# Residual plots suggest non-linearity. 
# Obs 14 has high leverage.

# (e)
model3a = lm(mpg~displacement*weight+year, data=auto)
model3b = lm(mpg~displacement+weight+origin:year, data=auto)
model3c = lm(mpg~displacement+acceleration*weight+year+origin, data=auto)
summary(model3a)
summary(model3b)
summary(model3c)

# (f)
model3d = lm(mpg~displacement+I(log(weight))+year, data=auto)
model3e = lm(mpg~displacement+I(sqrt(weight))+year+origin, data=auto)
# raw polynomial terms
model3fa = lm(mpg~displacement+I(poly(weight,2,raw=TRUE)), data=auto)
# orthogonal polynomial terms
model3fb = lm(mpg~displacement+I(poly(weight,2)), data=auto) 
summary(model3d)
summary(model3e)
summary(model3fa)
summary(model3fb)

# -------------------------------------------
# Q10
rm(list=ls())

library(ISLR)
fix(Carseats)

# (a)
model1 = lm(Sales~Price+Urban+US, data=Carseats)
summary(model1)

# (b)
# Sales drop by 0.054 for each dollar increase in Price.
# Sales are 0.022 lower for Urban areas.
# Sales are 1.201 higher in the US.

# (c)
# Sales = 13.043 - 0.054 x Price - 0.022 x UrbanYes + 1.201 x USYes

# (d)
# Price and USYes

# (e)
model2 = lm(Sales~Price+US, data=Carseats)
summary(model2)

# (f)
# model1: RSE = 2.472, adj.R2 = 0.2335 
# model2: RSE = 2.469, adj.R2 = 0.2354

# (g)
confint(model2)

# (h)
dev.new()
par(mfrow=c(2,2))
plot(model2)
# Residual plots look good, so no outliers. 

dev.off()
plot(hatvalues(model2))
which.max(hatvalues(model2))
# Data have leverage issues, particularly obs 43.

# -------------------------------------------
# Q11
rm(list=ls())

set.seed(1)
x = rnorm(100)
y = 2*x + rnorm(100)

# (a)
model1 = lm(y~x+0)
summary(model1)

# (b)
model2 = lm(x~y+0)
summary(model2)

# (c)
# model 1: y = ax
# model 2: x = by
# Thus, a = 1/b or b = 1/a.

# (d) & (e) skip

# (f)
model3 = lm(y~x)
model4 = lm(x~y)
summary(model3)
summary(model4)
# both t = 18.56

# -------------------------------------------
# Q12
# (a) skip
# (b) Q11 is an example
# (c)
rm(list=ls())

set.seed(2)
x = rnorm(100, mean=1000, sd=0.01)
y = rnorm(100, mean=1000, sd=0.01)

model5 = lm(y~x+0)
model6 = lm(x~y+0)
summary(model5)
summary(model6)

# -------------------------------------------
# Q13
rm(list=ls())
set.seed(1)

# (a)
x = rnorm(100, mean=0, sd=1)

# (b)
eps = rnorm(100, mean=0, sd=sqrt(0.25))

# (c)
y = -1 + 0.5*x + eps
length(y) # length = 100
# b0 = -1, b1 = 0.5

# (d)
plot(x,y) # positively correlated

# (e)
model1 = lm(y~x)
summary(model1)
# intercept and coefficient clost to b0 and b1 in part (c)

# (f)
dev.new()
plot(x,y) 
abline(model1, col='blue')
abline(-1, 0.5, col='red')
legend(1,-2, legend=c('model fit', 'population'), col=c('blue','red'), lwd=2)

# (g)
model2 = lm(y~poly(x,2))
summary(model2)
anova(model1, model2)
# No evidence on polynomial being better. 

# (h)-(j) skip

# -------------------------------------------
# Q14
rm(list=ls())

# (a)
set.seed(1)
x1 = runif(100)
x2 = 0.5*x1 + rnorm(100)/10
y  = 2 + 2*x1 + 0.3*x2 + rnorm(100)

# (b)
cor(x1,x2)
plot(x1,x2)

# (c)
model1 = lm(y~x1+x2)
summary(model1)
# reject null for b1, fail to reject null for b2

# (d)
model2 = lm(y~x1)
summary(model2)
# reject

# (e)
model3 = lm(y~x2)
summary(model3)
# reject

# (f): No, because we need both x1 and x2 in the same regression. 

# (g)
x1 = c(x1, 0.1)
x2 = c(x2, 0.8)
y  = c(y, 6)

model1 = lm(y~x1+x2)
summary(model1)
dev.new()
par(mfrow=c(2,2))
plot(model1)
# fail to reject null for b1, reject null for b2

model2 = lm(y~x1)
summary(model2)
dev.new()
par(mfrow=c(2,2))
plot(model2)
# reject
# new point high leverage for x1

model3 = lm(y~x2)
summary(model3)
dev.new()
par(mfrow=c(2,2))
plot(model3)
# reject
# new point outlier & high leverage for x2

# -------------------------------------------
# Q15
rm(list=ls())

library(MASS)
fix(Boston)

# (a)
pval_list = rep(0,13)
var_list  = names(Boston[-1])

for (i in 1:13) {
  x     = Boston[,i+1]
  model = lm(Boston$crim~x)

  f = summary(model)$fstatistic
  p = pf(f[1], f[2], f[3], lower.tail=F)
  attributes(p) = NULL
  pval_list[i]  = p 
}

rbind(var_list, pval_list)
# chas is the only insignificant predictor.

# (b)
model.all = lm(crim~., data=Boston)
summary(model.all)
# reject for: zn, nox, dis, rad, black, istat, medv

# (c)
uni_coef = rep(0,13)
for (i in 1:13) {
  x           = Boston[,i+1]
  model       = lm(Boston$crim~x)
  uni_coef[i] = coef(model)[2]
}

mul_coef = coef(model.all)[-1]

plot(uni_coef, mul_coef)
rbind(uni_coef, mul_coef)
# Coefficient estimates for nox are way off. 

# (d)
pval_pol2 = rep(0,13)
pval_pol3 = rep(0,13)

for (i in 1:13) {
  
  if (i != 3){ # skip chas cuz it's a factor variable  
    x     = Boston[,i+1]
    model = lm(Boston$crim~poly(x,3))
    
    pval_pol2[i] = summary(model)[["coefficients"]][, "Pr(>|t|)"][3]
    pval_pol3[i] = summary(model)[["coefficients"]][, "Pr(>|t|)"][4]
    }
 
}

rbind(var_list, pval_pol2, pval_pol3)
# No evidence of non-linear association for Black.
