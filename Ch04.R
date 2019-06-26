# Ch 4 Applied
# -------------------------------------------
# Q10
rm(list=ls())

library(ISLR)
fix(Weekly)

# (a)
summary(Weekly)
cor(Weekly[,-9])
dev.new()
pairs(Weekly)
# Volume increases in year.

# (b)
glm1.fit = glm(Direction ~ Lag1+Lag2+Lag3+Lag4+Lag5+Volume, data=Weekly, family=binomial)
summary(glm1.fit)
# Lag2 is statistically significant.

# (c)
contrasts(Weekly$Direction)
n         = dim(Weekly)[1] # # of obs
glm1.prob = predict(glm1.fit, type='response')
glm1.pred = rep('Down', n) 
glm1.pred[glm1.prob > .5] = 'Up'

# confusion matrix
table(glm1.pred, Weekly$Direction)
# When prediction is Down, model is right 54/(54+48) = 53%.
# When prediction is Up, model is right 557/(430+557)= 56%.

# Overall fraction of correct predictions
mean(glm1.pred == Weekly$Direction) # 56%

# (d)
train = (Weekly$Year < 2009) 
test  = Weekly[!train,] 
dim(test) # 104, 9

glm2.fit  = glm(Direction ~ Lag2, data=Weekly, family=binomial, subset=train)
glm2.prob = predict(glm2.fit, test, type='response')
glm2.pred = rep('Down', 104) 
glm2.pred[glm2.prob > .5] = 'Up'

# confusion matrix
table(glm2.pred, test$Direction)

# Overall fraction of correct predictions
mean(glm2.pred == test$Direction) # 62.5%

# (e) 
library (MASS) 
lda.fit   = lda(Direction ~ Lag2, data=Weekly, subset=train) 
lda.fit
lda.pred  = predict(lda.fit, test)
lda.class = lda.pred$class

table(lda.class, test$Direction)
mean(lda.class == test$Direction) # 62.5%

# (f) 
qda.fit   = qda(Direction ~ Lag2, data=Weekly, subset=train) 
qda.fit
qda.pred  = predict(qda.fit, test)
qda.class = qda.pred$class

table(qda.class, test$Direction)
mean(qda.class == test$Direction) # 58.7%

# (g)
library(class) 
train.X = as.matrix(Weekly$Lag2[train])
train.Y = as.matrix(Weekly$Direction[train])
test.X  = as.matrix(Weekly$Lag2[!train])

set.seed(1) 
knn.pred = knn(train.X, test.X, train.Y, k=1) 
table(knn.pred, test$Direction) # 50%

# (h): Logistic and LDA models are the best.

# (i)
# KNN
set.seed(1) 
knn.pred = knn(train.X, test.X, train.Y, k=5) 
table(knn.pred, test$Direction) # 53.8%
knn.pred = knn(train.X, test.X, train.Y, k=10) # best KNN
table(knn.pred, test$Direction) # 57.7%
knn.pred = knn(train.X, test.X, train.Y, k=20) 
table(knn.pred, test$Direction) # 56.7%

# LDA
lda.fit   = lda(Direction ~ Lag1*Lag2, data=Weekly, subset=train) 
lda.pred  = predict(lda.fit, test)
lda.class = lda.pred$class
mean(lda.class == test$Direction) # 57.7%

# QDA
qda.fit   = qda(Direction ~ Lag1*Lag2, data=Weekly, subset=train) 
qda.pred  = predict(qda.fit, test)
qda.class = qda.pred$class
mean(qda.class == test$Direction) # 46.2%

# -------------------------------------------
# Q11
rm(list=ls())
auto = read.csv("C:\\Users\\Carol\\Desktop\\Auto.csv", header=T, na.strings='?')
auto = na.omit(auto)

# (a)
mpg01   = ifelse(auto$mpg > median(auto$mpg), 1, 0)
newauto = data.frame(mpg01, auto)

# (b)
dev.new()
pairs(newauto) 
# displacement, horesepower, weight, acceleration

# (c)
set.seed(1)
id    = runif(length(mpg01))
train = (id < 0.8) # 80% of data
test  = newauto[!train,] # 20% of data
dim(test)

# (d)
lda.fit   = lda(mpg01 ~ displacement+horsepower+weight+acceleration, data=newauto, subset=train) 
lda.pred  = predict(lda.fit, test)
lda.class = lda.pred$class
mean(lda.class != test$mpg01) # 14.1%

# (e)
qda.fit   = qda(mpg01 ~ displacement+horsepower+weight+acceleration, data=newauto, subset=train) 
qda.pred  = predict(qda.fit, test)
qda.class = qda.pred$class
mean(qda.class != test$mpg01) # 16.9%

# (f)
glm.fit  = glm(mpg01 ~ displacement+horsepower+weight+acceleration, data=newauto, family=binomial, subset=train)
glm.prob = predict(glm.fit, test, type='response')
glm.pred = rep(0, dim(test)[1]) 
glm.pred[glm.prob > .5] = 1
mean(glm.pred != test$mpg01) # 12.7%

# (g)
train.X = cbind(newauto$displacement, newauto$horesepower, newauto$weight, newauto$acceleration)[train,]
train.Y = as.matrix(newauto$mpg01[train])
test.X  = cbind(newauto$displacement, newauto$horesepower, newauto$weight, newauto$acceleration)[!train,]

set.seed(1) 
knn.pred = knn(train.X, test.X, train.Y, k=1) 
mean(knn.pred != test$mpg01) # 19.7%
knn.pred = knn(train.X, test.X, train.Y, k=10) # best KNN
mean(knn.pred != test$mpg01) # 15.5%
knn.pred = knn(train.X, test.X, train.Y, k=20) 
mean(knn.pred != test$mpg01) # 16.9%
knn.pred = knn(train.X, test.X, train.Y, k=30) 
mean(knn.pred != test$mpg01) # 16.9%

# -------------------------------------------
# Q12
rm(list=ls())

# (a)
Power = function() {
  print(2^3)
}
Power()

# (b)
Power2 = function(x, a){
  print(x^a)  
}
Power2(3,8)

# (c)
Power2(10,3)  # 1000
Power2(8,17)  # 2.2518e+15
Power2(131,3) # 2248091

# (d)
Power3 = function(x, a){
  result = x^a
  return(result)
}

# (e)
x = c(1:10)
y = Power3(x,2)
dev.new()
par(mfrow=c(2,2))
plot(x, y, main='log(x^2) vs x', xlab='x', ylab='log(x^2)')
plot(x, y, log="x",  main='log(x^2) vs x on xlog-scale', xlab='x', ylab='log(x^2)')
plot(x, y, log="y",  main='log(x^2) vs x on ylog-scale', xlab='x', ylab='log(x^2)')
plot(x, y, log="xy", main='log(x^2) vs x on xylog-scale', xlab='x', ylab='log(x^2)')

# (f)
PlotPower = function(x,a){
  y = Power3(x,a)
  plot(x, y, main=paste('x^',a,'vs x'), xlab='x', ylab=paste('x^',a))
}
PlotPower(1:10,3)

# -------------------------------------------
# Q13
rm(list=ls())

library(MASS)
data(Boston)
crim01 = ifelse(Boston$crim > median(Boston$crim), 1, 0)
newdf  = data.frame(crim01, Boston)

dev.new()
pairs(newdf) # nox, rm, dis, tax, black, lstat, medv
cor(newdf)   # indus, nox, age, dis, rad, tax
# pick: indus, nox, dis, tax, lstat

set.seed(1)
id    = runif(length(crim01))
train = (id < 0.7)     # 70% of data for training
test  = newdf[!train,] # 30% of data for testing

# logit
glm.fit  = glm(crim01 ~ indus + nox + dis + tax + lstat, data=newdf, family=binomial, subset=train)
glm.prob = predict(glm.fit, test, type='response')
glm.pred = rep(0, dim(test)[1]) 
glm.pred[glm.prob > .5] = 1
mean(glm.pred != test$crim01) # 11% 

# LDA
lda.fit   = lda(crim01 ~ indus + nox + dis + tax + lstat, data=newdf, subset=train) 
lda.pred  = predict(lda.fit, test)
lda.class = lda.pred$class
mean(lda.class != test$crim01) # 16.4%

# QDA
qda.fit   = qda(crim01 ~ indus + nox + dis + tax + lstat, data=newdf, subset=train) 
qda.pred  = predict(qda.fit, test)
qda.class = qda.pred$class
mean(qda.class != test$crim01) # 15.1%

# KNN
train.X = cbind(newdf$indus, newdf$nox, newdf$dis, newdf$tax, newdf$lstat)[train,]
train.Y = newdf$crim01[train]
test.X  = cbind(newdf$indus, newdf$nox, newdf$dis, newdf$tax, newdf$lstat)[!train,]

library(class)
set.seed(1) 
knn.pred = knn(train.X, test.X, train.Y, k=1) # best KNN & overall 
mean(knn.pred != test$crim01) # 4.8%
knn.pred = knn(train.X, test.X, train.Y, k=10) 
mean(knn.pred != test$crim01) # 9.6%
knn.pred = knn(train.X, test.X, train.Y, k=20) 
mean(knn.pred != test$crim01) # 15.8%
knn.pred = knn(train.X, test.X, train.Y, k=30) 
mean(knn.pred != test$crim01) # 16.4%
knn.pred = knn(train.X, test.X, train.Y, k=50) 
mean(knn.pred != test$crim01) # 21.9%
knn.pred = knn(train.X, test.X, train.Y, k=100) 
mean(knn.pred != test$crim01) # 22.6%
knn.pred = knn(train.X, test.X, train.Y, k=200) 
mean(knn.pred != test$crim01) # 21.9%
knn.pred = knn(train.X, test.X, train.Y, k=300) 
mean(knn.pred != test$crim01) # 21.9%
