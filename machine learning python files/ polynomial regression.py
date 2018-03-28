##polynomial regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values  ## make x as matrix 
Y = dataset.iloc[:,2].values ## make y as vector

## fitting Linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

## fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3) ##change the degree to make the curve more predictive
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,Y)

## Visualizing the linear Regression results
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('linear regression ')
plt.xlabel("Position level")
plt.ylabel('Position Salary')
plt.plot()
 
##Visualizing the polynomial regression results
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Polynomial regression ')
plt.xlabel("Position level")
plt.ylabel('Position Salary')
plt.plot()


##Visualizing the polynomial regression results with added values
X_val = np.arange(min(X),max(X),0.1)
X_val= X_val.reshape((len(X_val),1))    
plt.scatter(X,Y,color='red')
plt.plot(X_val,lin_reg2.predict(poly_reg.fit_transform(X_val)),color='blue')
plt.title('Polynomial regression ')
plt.xlabel("Position level")
plt.ylabel('Position Salary')
plt.plot()


##Predicting the new result with linear regression 
lin_reg.predict(6.5)

##Predicting the new result with polynomial regression 
lin_reg2.predict(poly_reg.fit_transform(6.5))