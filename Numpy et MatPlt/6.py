import math
import numpy as np                      # charge un package pour le numérique
import matplotlib.pyplot as plt         # charge un package pour les graphiques 

centroids_train = []
for i in range(10):
    X_train_i = X_train[y_train == i]
    centroids_train.append(np.mean(X_train_i, axis=0))