# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 12:52:25 2019

@author: benoi
"""
#import modules
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns


#import dataset
dataset = pd.read_csv('A2Z_Insurance.csv')


###############################################################################
###############  DATA PREPROCESSING  ##########################################
###############################################################################

#set index with cutomer id
dataset = dataset.set_index("Customer Identity")

#deleting geographic area bc no further info avaiblable
dataset = dataset.drop("Geographic Living Area",1)

#cleaning the column educational degree 
dataset['Educational Degree'] = dataset['Educational Degree'].str.replace(' ','').str.replace('H','').str.replace('i','').str.replace('g','').str.replace('h','').str.replace('S','').str.replace('c','').str.replace('o','').str.replace('l','').str.replace('-','').str.replace('B','').str.replace('M','').str.replace('/','').str.replace('s','').str.replace('P','').str.replace('D','').str.replace('a','')

#extend dataset with Age column
year = 2019
Born_year = dataset["Brithday Year"]
Age = year - Born_year
Age = pd.DataFrame(Age)
Age.columns = ['age']
#merge on index
dataset= pd.merge(Age,dataset, right_index=True, left_index=True)

#deleting space in columns name
dataset = dataset.rename(columns={'First Policy´s Year':'FirstPolicyYear','Customer Monetary Value':'CustomerMonetaryValue','Gross Monthly Salary':'GrossMonthlySalary'})

#float to int 
dataset['CustomerMonetaryValue'] = dataset.CustomerMonetaryValue.astype(int)

#filtering impossible values for salary and CMV




###############################################################################
#######################   FUNCTIONS   #########################################
###############################################################################

def count_numbers_in_list (age_category):
    number = len(age_category)
    return number


def split_in_age_category (dataset):
    
    year = 2019
    Born_year = dataset["Brithday Year"]
    Age = year - Born_year
    
    age_18andless =[]
    age_18_24 = []
    age_25_34 = []
    age_35_44 = []
    age_45_54 = []
    age_55_64 = []
    age_65andmore = []
    invalid = []
    
    for value in Age: 
        if value < 18:
            age_18andless.append(value)
        elif value >= 18 and value <= 24: 
            age_18_24.append(value) 
        elif value >= 25 and value <= 34: 
            age_25_34.append(value)
        elif value >= 35 and value <= 44:
            age_35_44.append(value)
        elif value >= 45 and value <= 54:
            age_45_54.append(value)
        elif value >= 55 and value <= 64:
            age_55_64.append(value)
        elif value >= 65:
            age_65andmore.append(value)
        else: 
            invalid.append(value)

    lst = [age_18_24, age_25_34, age_35_44, age_45_54, age_55_64, age_65andmore]
    number_of_customers = []
    
    for category in lst:
        number = count_numbers_in_list(category)
        number_of_customers.append(number)

    Age_dataframe = pd.DataFrame(list(number_of_customers))
    category_name = ['age18-24', 'age25-34','age35-44','age45-54', 'age55-64','age65+']
    Age_dataframe["age_category"] = category_name
    Age_dataframe = Age_dataframe.rename(columns={0:'total'})
    
    return Age_dataframe


###############################################################################
####################   ANALYSING ALL AGES CATEGORY  ###########################
###############################################################################

Age_dataframe_all_dataset = split_in_age_category(dataset)
Age_dataframe_all_dataset.plot(kind='bar',x='age_category',y='total',  title='')
plt.xlabel('Age category')
plt.ylabel('Number of Customers')
plt.show()

statistics_all_customers = dataset.describe()


###############################################################################
####################   ANALYSING THE CATEGORY 65+   ###########################
###############################################################################

#second df
cols = [6,7,8,9,10,11]
new_df = dataset.copy()
new_df.drop(new_df.columns[cols],axis=1,inplace=True)

#get the id of all customers that are in the category 65+
values = range(65,100)
lst_values = []
for i in values :
    lst_values.append(i)

#df with only the 65+
df_65 = dataset.loc[dataset['age'].isin(list(lst_values))]

#correlation
df_65_corr = df_65.copy() 
df_65_corr = df_65.corr()
f, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(df_65_corr, mask=np.zeros_like(df_65_corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)

#verification df_65 contains only 65 + customers
Age_dataframe_65 = split_in_age_category(df_65)
Age_dataframe_65.plot(kind='bar',x='age_category',y='total',  title='')
plt.xlabel('Age category')
plt.ylabel('Number of Customers')
plt.show()


###############################################################################
####################   First policy year && customer monetary value  ANALYSIS #
###############################################################################


dataset_FPY_CMV = dataset.copy()

dataset_sample = dataset_FPY_CMV[:250]
dataset_sample.plot(kind='scatter',x='FirstPolicyYear',y='CustomerMonetaryValue',color='red')
plt.xlim(1970,2000 )
plt.ylim(-100, 600)



#Best_customers_df 
CMV_values = range(200,600)
Best_CMV_values = []
for i in CMV_values:
    Best_CMV_values.append(i)
    
FPY_values = range(1970,1985)
Best_FPY_values = []
for i in FPY_values:
    Best_FPY_values.append(i)

df_Best_CMV_Customers = dataset_FPY_CMV.loc[dataset_FPY_CMV['CustomerMonetaryValue'].isin(list(Best_CMV_values))]
df_Best_FPY_Customers = dataset_FPY_CMV.loc[dataset_FPY_CMV['FirstPolicyYear'].isin(list(Best_FPY_values))]

index_Best_CMV = df_Best_CMV_Customers.index.values.tolist() 
index_Best_FPY = df_Best_FPY_Customers.index.values.tolist()

#LIST WITH THE BEST CUSTOMERS CMV &&& FPY
new_list = []
for element in index_Best_CMV:
    if element in index_Best_FPY:
        new_list.append(element)

#df with best customers CMV & FPY
df_Best_Customers = dataset_FPY_CMV.loc[dataset_FPY_CMV.index.isin(list(new_list))]    
df_Best_Customers.plot(kind='scatter',x='FirstPolicyYear',y='CustomerMonetaryValue',color='red')
plt.xlim(1970,2000 )
plt.ylim(-100, 600)

# Best customers df  analysis
df_Best_Customers_corr = df_Best_Customers.corr()
statistics_best_customers = df_Best_Customers.describe()

Age_dataframe_Best_customers = split_in_age_category(df_Best_Customers)
Age_dataframe_Best_customers.plot(kind='bar',x='age_category',y='total',  title='CMV&PDY')
plt.xlabel('Age category')
plt.ylabel('Number of Customers')
plt.show()

###############################################################################
###########################   Clustering ######################################
###############################################################################


############################## K-MEANS ########################################


#Picking the 2 variables that we are interst in 
X = dataset.iloc[:, [4, 6]].values

# Taking care of missing data NaN
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:5])
X[:, 0:5] = imputer.transform(X[:, 0:5])

#float to int 
#dataset['CustomerMonetaryValue'] = dataset.CustomerMonetaryValue.astype(int)

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)


# Visualising the clusters ONLY FOR 2D DIMENSIONAL CLUSTERS!!!!!!!!!!!!
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Gros Monthly Salary')
plt.ylabel('CMV')
plt.legend()
plt.show()


############################ HIERARCHICAL CLUSTERING ##########################

'''
# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
'''


# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
# WARD = method that minimize the deviation in each cluster 
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Gross Monthly Salary')
plt.ylabel('CMV')
plt.legend()
plt.show()







# Tgeriuh82yo40pyt84tghio









'''












