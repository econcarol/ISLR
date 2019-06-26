# Ch 2 Applied
# -------------------------------------------
# Q8
rm(list=ls())

# (a)
college = read.csv("C:\\Users\\Carol\\Desktop\\College.csv", header=T)

# (b)
rownames(college) = college[,1] 
college = college[,-1]
fix(college)

# (c)
# (i)
summary(college)

# (ii)
dev.new()
pairs(college[,1:10])

# (iii)
dev.new()
plot(college$Private, college$Outstate, xlab='Private', ylab='Out-of-state tuition')

# (iv)
Elite = rep("No", nrow(college)) 
Elite[college$Top10perc > 50] = "Yes" 
Elite = as.factor(Elite) 
college = data.frame(college, Elite)

summary(college$Elite) # 78 Elite
plot(college$Elite, college$Outstate, xlab='Elite',ylab='Out-of-state tuition')

# (v)
dev.new()
par(mfrow=c(2,2)) 
hist(college$Apps, breaks=50, main='# of applications')
hist(college$Enroll, breaks=25, main='# of new enrollment')
hist(college$Expend, breaks =10, main='Instructional expenditure per student')
hist(college$Outstate, main='Out-of-state tuition')

# -------------------------------------------
# Q9
rm(list=ls())

# (a)
auto = read.csv("C:\\Users\\Carol\\Desktop\\Auto.csv", header=T, na.strings='?')
auto = na.omit(auto)
fix(auto)
summary(auto) # qualitative: origin, name

# (b)
sapply(auto[,1:7], range)

# (c)
sapply(auto[,1:7], mean)
sapply(auto[,1:7], sd)

# (d)
sapply(auto[-(10:85),1:7], range)
sapply(auto[-(10:85),1:7], mean)
sapply(auto[-(10:85),1:7], sd)

# (e)
dev.new()
pairs(auto[,1:7])

# (f)
# based on (e):
# mpg is negatively correlated with cylinders, displacement, horsepower, and weight
# mpg is positively correlated with acceleration and year

# -------------------------------------------
# Q10
rm(list=ls())

# (a)
library(MASS)
?Boston
dim(Boston) # 506 rows, 14 cols
fix(Boston) # row: obs, col: variables

# (b)
dev.new()
pairs(Boston)

# (c)
pairs(Boston$crim~Boston$zn)
pairs(Boston$crim~Boston$indus)
pairs(Boston$crim~Boston$chas)
pairs(Boston$crim~Boston$nox)
pairs(Boston$crim~Boston$rm)
pairs(Boston$crim~Boston$age)
pairs(Boston$crim~Boston$dis)
pairs(Boston$crim~Boston$rad)
pairs(Boston$crim~Boston$tax)
pairs(Boston$crim~Boston$ptratio)
pairs(Boston$crim~Boston$black)
pairs(Boston$crim~Boston$lstat)
pairs(Boston$crim~Boston$medv)

# (d)
dev.new()
par(mfrow=c(3,1)) 
plot(Boston$crim)
plot(Boston$tax)
plot(Boston$ptratio)

range(Boston$crim)
range(Boston$tax)
range(Boston$ptratio)

# (e)
table(Boston$chas) # 35 towns

# (f)
median(Boston$ptratio) # 19.05 pupils per teacher

# (g)
which(Boston$medv == min(Boston$medv)) # town 399 and 406
rbind(sapply(Boston[,2:14], min), 
      Boston[Boston$medv == min(Boston$medv), 2:14], 
      sapply(Boston[,2:14], max))

# (h)
nrow(Boston[Boston$rm > 7,]) #64
nrow(Boston[Boston$rm > 8,]) #13
rbind(sapply(Boston[Boston$rm > 8,], mean), sapply(Boston, median))