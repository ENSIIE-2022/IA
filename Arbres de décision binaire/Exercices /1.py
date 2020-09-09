import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from graphviz import Source
from sklearn import tree

dataset = pd.read_csv("Wine_grading.csv", index_col=0)
dataset.head()

#Toutes les colonnes sauf la dernière : grade
X = dataset.iloc[:, :-1]
#La colonne grade
y = dataset.iloc[:, -1]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
label_encoder.classes_

sc_x = StandardScaler()
X = sc_x.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

trains = [[],[],[]]
tests = [[],[],[]]

for i in range(1,500):
    classifier = DecisionTreeClassifier(max_depth=i)
    classifier.fit(X_train, y_train)

    trains[0].append(format(classifier.score(X_train, y_train)))
    tests[0].append(format(classifier.score(X_test, y_test)))
    
#Source(tree.export_graphviz(classifier, out_file=None, feature_names=dataset.columns[:-2], filled = True))
print("Train Score (indices => values) : "+str((trains[0].index(max(trains[0]))))+" => "+max(trains[0]))
print("Test Score (indices => values) : "+str((tests[0].index(max(tests[0]))))+" => "+max(tests[0]))

for i in range(100,500):
    classifier = RandomForestClassifier(n_estimators=i)
    classifier.fit(X_train, y_train)

    trains[1].append(format(classifier.score(X_train, y_train)))
    tests[1].append(format(classifier.score(X_test, y_test)))

print("Train Score (indices => values) : "+str((trains[1].index(max(trains[1]))))+" => "+max(trains[1]))
print("Test Score (indices => values) : "+str((tests[1].index(max(tests[1]))))+" => "+max(tests[1]))


for i in range(2,60):
    for j in range(100,150):
        classifier = RandomForestClassifier(n_estimators = j, min_samples_split=i)
        classifier.fit(X_train, y_train)

        trains[2].append(format(classifier.score(X_train, y_train)))
        tests[2].append(format(classifier.score(X_test, y_test)))

print("Train Score (indices => values) : "+str((trains[2].index(max(trains[2]))))+" => "+max(trains[2]))
print("Test Score (indices => values) : "+str((tests[2].index(max(tests[2]))))+" => "+max(tests[2]))

#cm = confusion_matrix(y_test, classifier.predict(X_test))
#sns.heatmap(cm, annot=True, fmt=".1f")

#feature_importance = pd.DataFrame({"Features":dataset.columns[:-1], "Value": classifier.feature_importances_})
#feature_importance.sort_values(["Value"], ascending=False)


