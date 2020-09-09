import numpy as np                      # charge un package pour le numérique
import matplotlib.pyplot as plt         # charge un package pour les graphiques 

#récupérer dans une variable X_zero tous les vecteur de cette classe
X_zero = X[y==1]
#afficher le nombre de vecteur
print(X_zero.shape[0])
#calculer dans une variable X_zero_mean le vecteur moyen
X_zero_mean = np.mean(X_zero, axis =0)


img = np.reshape(X_zero_mean, (8,8))

plt.imshow(img, cmap='gray', aspect='equal', interpolation='nearest')