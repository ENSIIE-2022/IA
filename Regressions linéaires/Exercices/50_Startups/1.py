import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

dataframe = pd.read_csv("50_Startups.csv")
print(dataframe.head())

for column in [x for x in dataframe.columns if x not in ['Profit','State']]:
    sns.catplot(x=column, y="Profit", data=dataframe)

    X = dataframe[[column]]
    y = dataframe[["Profit"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    regressor.coef_
    regressor.intercept_
    regressor.score(X_train, y_train)

    print("["+column+"] Score de Train : {}".format(regressor.score(X_train, y_train)))
    print("["+column+"] Score de Test : {}".format(regressor.score(X_test, y_test)))

    regressor.predict(X_train)

    xp = 200

    # Visualize our Training Set
    plt.scatter(X_train, y_train, color="red")
    plt.plot(X_train, regressor.predict(X_train), color="blue")
    plt.title("Profit VS "+column+" (training set)")
    plt.xlabel(column)
    plt.ylabel("Profit")
    plt.show()

    ## Visualize our Test Set
    #plt.scatter(X_test, y_test, color="red")
    #plt.plot(X_test, regressor.predict(X_test), color="blue")
    #plt.title("Profit VS "+column+" (test set)")
    #plt.xlabel(column)
    #plt.ylabel("Profit")
    #plt.show()