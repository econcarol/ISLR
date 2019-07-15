# Ch 10 Applied
# -------------------------------------------
# Q7
rm(list=ls())

fix(USArrests)
sd.df = scale(USArrests)

Edis  = as.matrix(dist(t(sd.df), method='euclidean'))
Edis2 = Edis^2
corr  = cor(sd.df)
prop  = (1-corr)/Edis2 #0.0102

# -------------------------------------------
# Q8
# (a)
pr.out = prcomp(sd.df, scale=F)
pr.var = pr.out$sdev^2
pve1   = pr.var/sum(pr.var)
pve1

# (b)
pve2 = rep(0, 4)
for (i in 1:4){
  phi     = t(as.matrix(pr.out$rotation[,i]))
  pve2[i] = sum((phi%*%t(sd.df))^2)/sum(sd.df^2)
}
pve2

# -------------------------------------------
# Q9
# (a)
hc.complete1 = hclust(dist(USArrests), method='complete')

# (b)
hc.3clusters1  = cutree(hc.complete1, 3)
res1           = data.frame(hc.3clusters1)
colnames(res1) = 'id_or'

# (c)
hc.complete2   = hclust(dist(sd.df), method='complete')
hc.3clusters2  = cutree(hc.complete2, 3)
res2           = data.frame(hc.3clusters2)
colnames(res2) = 'id_sd'

# (d)
res = merge(res1, res2, by=0, all=T)
colnames(res) = c('state', 'id_or', 'id_sd')

dev.new()
par(mfrow=c(2,1), oma=c(0,0,2,0))
plot(res1$id, main='original data', xlab='', ylab='cluster ID')
plot(res2$id, main='scaled data', xlab='', ylab='cluster ID', col='blue')
mtext('complete linkage, 3 clusters', outer=T)

dev.new()
plot(hc.complete1, labels=row.names(USArrests), main='original data', sub='', xlab='', ylab='') 
abline(h=150, col="red")

dev.new()
plot(hc.complete2, labels=row.names(sd.df), main='scaled data', sub='', xlab='', ylab='') 
abline(h=4.4, col="red")

# Before scaling, cluster sizes are more equal. 
# Scaling puts more states in one particular cluster.  
# Scaling is better cuz crime frequencies and variable units are different. 

# -------------------------------------------
# Q10
rm(list=ls())

# (a)
n = 60
p = 50

set.seed(1)
y = c(rep(1,20), rep(2,20), rep(3,20))
x = matrix(rnorm(n*p), ncol=p)
x[1:20, 1:20]   = x[1:20, 1:20] + 3
x[21:40, 21:40] = x[21:40, 21:40] - 4

# (b)
pr.out = prcomp(x)

Cols = function(vec){
  cols = rainbow(length(unique(vec)))
  return(cols[as.numeric(as.factor(vec))])
}

dev.new()
plot(pr.out$x[,1:2], col=Cols(y), pch=19, xlab='Z1',ylab='Z2')

# (c)
set.seed(2)
km3.out = kmeans(x, 3, nstart =20)
km3.id  = km3.out$cluster
table(y, km3.id)
# Kmeans correctly identifies 3 clusters. 

# (d)
set.seed(3)
km2.out = kmeans(x, 2, nstart=20)
km2.id  = km2.out$cluster
table(y, km2.id)
# Kmeans assign y1 and y3 to the same cluster.

# (e)
set.seed(4)
km4.out = kmeans(x, 4, nstart=20)
km4.id  = km4.out$cluster
table(y, km4.id)
# Kmeans splits y1 into 2 different clusters.

# (f)
set.seed(5)
km3pr.out = kmeans(pr.out$x[,1:2], 3, nstart=20)
km3pr.id  = km3pr.out$cluster
table(y, km3pr.id)
# Kmeans correctly identifies 3 clusters.

# (g)
sd.x = scale(x)
set.seed(6)
km3sd.out = kmeans(sd.x, 3, nstart=20)
km3sd.id  = km3sd.out$cluster
table(y, km3sd.id)
# Kmeans correctly identifies 3 clusters.

# -------------------------------------------
# Q11
rm(list=ls())

# (a) 
df = read.csv('C:/Users/Carol/Desktop/Ch10Ex11.csv', header=F)
df = t(df)

# (b)
dd          = as.dist(1-cor(t(df)))
hc.complete = hclust(dd, method ='complete')
hc.average  = hclust(dd, method ='average')
hc.single   = hclust(dd, method ='single')

dev.new()
plot(hc.complete, main='complete linkage w/ correlation-based distance', xlab='', sub='')
dev.new()
plot(hc.average, main='avreage linkage w/ correlation-based distance', xlab='', sub='')
dev.new()
plot(hc.single, main='single linkage w/ correlation-based distance', xlab='', sub='')

cutree(hc.complete, 2) # identify the decreased group correctly 
cutree(hc.average, 2) # identify the decreased group correctly
cutree(hc.single, 2) # fail to identify 2 groups

# Yes, my results depend on the type of linkage used. 

# (c)
pr.out = prcomp(df)
pve    = pr.out$sdev^2/sum(pr.out$sdev^2)*100

dev.new()
par(mfrow=c(1,2))
plot(pve, type='o', ylab='PVE', xlab='principal component', col='blue')
plot(cumsum(pve), type='o', ylab='Cumulative PVE', xlab='principal component', col='brown3')
# There is an elbow after 1st principal component. 

# Top 5 genes that differ the most across 2 groups.
PC1.order5 = order(abs(pr.out$rotation[,'PC1']), decreasing=T)[1:5]
PC1.order5
