
# import the libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# read the dataset
dataset=pd.read_csv(r'C:\Users\ADMIN\Downloads\23rd- Poly\23rd- Poly\1.POLYNOMIAL REGRESSION\emp_sal.csv')

# x and y variables
x=dataset.iloc[: , 1:2].values
y = dataset.iloc[:,2].values


from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=2,algorithm='brute',leaf_size=100,p=1,weights="distance")
knn_reg.fit(x,y)

y_pred_knn = knn_reg.predict([[6.5]])
y_pred_knn

