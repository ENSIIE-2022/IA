import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

dataframe = pd.read_csv("Marketing_Spending.csv")
print(dataframe.head())

sns.catplot(x="Marketing Spending", y="Profit", data=dataframe)

X = dataframe[["Marketing Spending"]]
y = dataframe[["Profit"]]

#sc_X = StandardScaler()
#X = sc_X.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
regressor.coef_
regressor.intercept_
regressor.score(X_train, y_train)

print("Score de Train : {}".format(regressor.score(X_train, y_train)))
print("Score de Test : {}".format(regressor.score(X_test, y_test)))

"""
scores = []
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = i)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    scores.append(regressor.score(X_test, y_test))

scoresnp = np.array(scores)
print(scoresnp.mean())
print(scoresnp.std())
"""

regressor.predict(X_train)

xp = 200
#xp_normalized = 
#xp_normalized = sc_X.transform([[xp]])
#print("Predicted Profit: {}".format(regressor.predict(xp_normalized)))
#print("Predicted Marketing spending: {}".format(regressor.predict(np.array([[xp]]))))

# Visualize our Training Set
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Profit VS Marketing Spending (training set)")
plt.xlabel("Marketing Spending")
plt.ylabel("Profit")
plt.show()

## Visualize our Test Set
plt.scatter(X_test, y_test, color="red")
plt.plot(X_test, regressor.predict(X_test), color="blue")
plt.title("Profit VS Marketing Spending (test set)")
plt.xlabel("Marketing Spending")
plt.ylabel("Profit")
plt.show()