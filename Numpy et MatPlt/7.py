import math
import numpy as np                      # charge un package pour le numérique
import matplotlib.pyplot as plt         # charge un package pour les graphiques 

def getclosestcentroid(x, centroids_train):
    currentdist = np.inf
    for classe, vect_mean in enumerate(centroids_train):
        newdist = dist(x,vect_mean)
        if newdist<currentdist:
            candidate = vect_mean
            candidate_class = classe
            currentdist = newdist
    return candidate_class

#ok nous allons maintenant faire une prediction en se basant sur notre distance pour tous les éléments de X_Test
#initialisons y_pred
y_pred = np.zeros(len(y_test), dtype=int)

#faisons les prédiction
for i in range(X_test.shape[0]):
    x = X_test[i]
    y_pred[i] = getclosestcentroid(x, centroids_train)

#print(y_pred)
#print(y_test)
#notre % de réussite est le nombre moyen d'occurence commune
print('score :')
print(np.mean(y_pred == y_test))