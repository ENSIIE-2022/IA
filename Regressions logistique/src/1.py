import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm

dataset = pd.read_csv("Social_Network_Ads.csv")

dataset

sns.catplot(x="Gender", y="Purchased", data=dataset, kind="bar")
sns.lmplot(x="EstimatedSalary", y="Purchased", data=dataset, logistic=True)

dataset = dataset.drop(['User ID'], axis=1)

X = dataset.drop(["Purchased"], axis=1)
y = dataset["Purchased"]

X = pd.get_dummies(X, drop_first=True)
X.head()

sc_x = StandardScaler()
X = sc_x.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

print("Train score : {}".format(classifier.score(X_train, y_train)))
print("Test score : {}".format(classifier.score(X_test, y_test)))

classifier.predict_proba(X_test)

cm = confusion_matrix(y_test, classifier.predict(X_test))
sns.heatmap(cm, annot=True, fmt=".2f")
plt.ylabel('Y')
plt.xlabel('X')
plt.show()

classifier.coef_.squeeze()
pd.DataFrame({"Features": ["Age", "Estimated Salary", "Gender"], "Values": classifier.coef_.squeeze()})

logit_model = sm.Logit(y, sm.add_constant(X)).fit()
print(logit_model.summary())