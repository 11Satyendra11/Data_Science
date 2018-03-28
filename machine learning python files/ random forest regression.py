## Random forest regression.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2]
Y= dataset.iloc[:,2]

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,Y)

ypred = regressor.predict(6.5)

X_grid=  np.arange(min(X),max(X),0.1)  ## 0.01 makes the line more vertical
X_grid = X_grid.reshape(len(X_grid),1) 
plt.scatter(X,Y,color= 'red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('random forest regression ')
plt.xlabel('position level')
plt.ylabel('salary')
plt.plot()
