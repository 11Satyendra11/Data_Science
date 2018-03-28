## Support Vector Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2]
Y= dataset.iloc[:,2:3]

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

Y = np.reshape(Y,10)

from sklearn.svm import SVR  ##does not have feature scaling 
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)

ypred =sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))


plt.scatter(X,Y,color = 'red')
plt.plot(X,regressor.predict(X),color = 'blue')
plt.title('SVR')
plt.xlabel('Position level')
plt.ylabel('Salary')