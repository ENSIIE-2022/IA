import math
import numpy as np                      # charge un package pour le numérique
import matplotlib.pyplot as plt         # charge un package pour les graphiques 

X1, X2 = X[100], X[1000]

#écrire une fonction qui renvoie la distance l2 de deux éléments
def dist(X1,X2):
    return math.sqrt(np.sum((X1-X2)**2))