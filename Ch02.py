# ISLR Ch 2 by Carol Cui

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------
# Q8
# (a)
college = pd.read_csv('C:\\Users\\Carol\\Desktop\\college.csv') # import data
pd.options.display.max_rows = 10 # set max # of rows for display
college # show data

# (b)
school_name = college.iloc[:,0] # abstract a list of school names
# rename index of each row to its corresponding school name
college = college.rename(index = lambda x: school_name[x]) 
# drop the column of school name from the original data frame
college.drop(college.columns[[0]], axis=1, inplace=True)
# convert Private from string to categorical
college['Private'] = college.Private.astype('category')

# (c)
# (i)
college.describe()

# (ii)
pd.tools.plotting.scatter_matrix(college.iloc[:,0:10], alpha=0.2, figsize=(10,10))

# (iii)
sns.boxplot(x='Private', y='Outstate', data=college)

# (iv)
college['Elite'] = np.where(college['Top10perc'] > 50, 'Yes', 'No')
college['Elite'] = college.Elite.astype('category')

pd.crosstab(index=college['Elite'], columns='Count')  
sns.boxplot(x='Elite', y='Outstate', data=college)

# (v)
fig = plt.figure() 
fig.add_subplot(2, 2, 1)
plt.hist(college['Apps'], 50, facecolor='blue')
plt.xlabel('new applications')
fig.add_subplot(2, 2, 2)
plt.hist(college['Enroll'], 45, facecolor='blue')
plt.xlabel('new enrollment')
fig.add_subplot(2, 2, 3)
plt.hist(college['Expend'], 30, facecolor='blue')
plt.xlabel('Instructional expenditure per student')
fig.add_subplot(2, 2, 4)
plt.hist(college['Outstate'], facecolor='blue')
plt.xlabel('Out-of-state tuition')

# ----------------------------------------------------------------------------
# Q9
# (a)
# import data: treat ? as missing value
Auto = pd.read_csv('C:\\Users\\Carol\\Desktop\\Auto.csv', na_values='?') 
Auto = Auto.dropna() # drop all missing values
Auto.head()

Auto.dtypes # qualitative: name

# (b) & (c)
Auto.describe()

# (d)
temp = Auto.copy()
temp['remove'] = 0
for i in range(len(temp.index)):
    if i > 8 and i < 85:
        temp.iloc[i,9] = 1
# check
temp.iloc[8:87] 
temp = temp[temp.remove == 0]
temp.describe()

# (e) & (f)
pd.tools.plotting.scatter_matrix(Auto.iloc[:,0:7], alpha=0.5, figsize=(6,6))

# mpg is negatively correlated with cylinders, displacement, horsepower, and weight
# mpg is positively correlated with acceleration and year

# ----------------------------------------------------------------------------
# Q10
# (a)
Boston = pd.read_csv('C:\\Users\\Carol\\Desktop\\Boston.csv') 
# CSV data comes with 1st column indicating suburb #'s,
# so we change the default index # to the info provided by this column
index  = Boston.iloc[:,0]
Boston = Boston.rename(index = lambda x: index[x]) 
Boston.drop(Boston.columns[[0]], axis=1, inplace=True)
Boston.head()

Boston.shape 
# 506 rows, 14 columns. row: obs, col: variables.

# (b)
pd.tools.plotting.scatter_matrix(Boston.iloc[:,0:13], alpha=0.5, figsize=(13,13))

# (c)
fig = plt.figure() 
for i in range(1, 14, 1):
    fig.add_subplot(3, 5, i)
    plt.scatter(Boston.iloc[:,i], Boston.iloc[:,0])
    plt.xlabel(Boston.columns[i])
    plt.ylabel(Boston.columns[0])

# (d)
fig = plt.figure()
fig.add_subplot(1, 3, 1)
plt.plot(Boston.crim)
plt.xlabel('suburb #')
plt.ylabel('crime rate per capita')
fig.add_subplot(1, 3, 2)
plt.plot(Boston.tax)
plt.xlabel('suburb #')
plt.ylabel('tax rate')
fig.add_subplot(1, 3, 3)
plt.plot(Boston.ptratio)
plt.xlabel('suburb #')
plt.ylabel('pupil-teacher ratio')

print('crime rate per capita: [%.1f, %.1f]'% (min(Boston.crim), max(Boston.crim)))
print('tax rate: [%.1f, %.1f]'% (min(Boston.tax), max(Boston.tax)))
print('pupil-teacher ratio: [%.1f, %.1f]'% (min(Boston.ptratio), max(Boston.ptratio)))

# (e)
pd.crosstab(index=Boston["chas"], columns="count") # 35

# (f)
Boston.ptratio.median() # 19.05

# (g)
town_index = [i for i, v in enumerate(Boston.medv) if v == Boston.medv.min()]
print(town_index+1) # 399, 406

minv  = [Boston.iloc[:,i].min() for i in range(1, 14, 1)]
minv  = np.reshape(minv, (1,13))
maxv  = [Boston.iloc[:,i].max() for i in range(1, 14, 1)]
maxv  = np.reshape(maxv, (1,13))
town1 = [Boston.iloc[town_index[0], 1:14]]
town2 = [Boston.iloc[town_index[1], 1:14]]
df1   = np.stack((minv, town1, town2, maxv))
print(pd.DataFrame(df1[:,0,:], index=['min','town1','town2','max'], columns=Boston.columns[1:14]))

# (h)
sum(Boston.rm > 7) # 64
sum(Boston.rm > 8) # 13

Boston['rm8'] = np.where(Boston['rm'] > 8, True, False)
mean8         = [Boston.groupby(['rm8'])[Boston.columns[i]].mean()[1] for i in range(0, 14, 1)]
medv          = [Boston.iloc[:,i].median() for i in range(0, 14, 1)]
df2           = np.stack((mean8, medv))
print(pd.DataFrame(df2, index=['rm>8 mean','data median'], columns=Boston.columns[0:14]))
