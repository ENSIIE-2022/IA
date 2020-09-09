import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

dataset = pd.read_csv("californian_housing.csv", index_col=0)
dataset.head()

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

dataset['PriceInt'] = dataset['Price'].apply(lambda x : list(y.unique()).index(x))

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

label_encoder.classes_

sc_x = StandardScaler()
X = sc_x.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = DecisionTreeClassifier(min_samples_leaf=50)
classifier.fit(X_train, y_train)

print("Train Score : {}".format(classifier.score(X_train, y_train)))
print('Test Score : {}'.format(classifier.score(X_test, y_test)))

classifier = RandomForestClassifier(n_estimators=200)
classifier.fit(X_train, y_train)

print("Train Score : {}".format(classifier.score(X_train, y_train)))
print('Test Score : {}'.format(classifier.score(X_test, y_test)))

classifier = RandomForestClassifier(n_estimators = 150, min_samples_split=30)
classifier.fit(X_train, y_train)

print("Train Score : {}".format(classifier.score(X_train, y_train)))
print('Test Score : {}'.format(classifier.score(X_test, y_test)))

cm = confusion_matrix(y_test, classifier.predict(X_test))
sns.heatmap(cm, annot=True, fmt=".1f")

#feature_importance = pd.DataFrame({"Features":dataset.columns[:-1], "Value": classifier.feature_importances_})
#feature_importance.sort_values(["Value"], ascending=False)