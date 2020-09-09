import numpy as np                      # charge un package pour le numérique
import matplotlib.pyplot as plt         # charge un package pour les graphiques 

print(X.dtype)
print(y.dtype)
#print(X.shape) 1797, 64
#print(y.shape) 1797,

print("Nombre de pixels : "+str(X.size))

print("Nombre d'observations : "+str(y.size))

print("Nombre de classes : "+str(np.unique(y).size))

idx_to_test = 532
print(X[idx_to_test])
print(y[idx_to_test])

# Utilisation de la fonction imshow pour l'affichage de l'image numéro idx_to_test:
plt.imshow(np.reshape(X[idx_to_test, :], (8, 8)))

# Amélioration de la visualisation (niveau de gris) et de la légende:
plt.imshow(np.reshape(X[idx_to_test, :], (8, 8)), cmap='gray', aspect='equal', interpolation='nearest')

# Attention aux accents: ne pas oublier le u (Unicode) ci-dessous
plt.title(u'Le chiffre d\'indice %s est un %s' % (idx_to_test, y[idx_to_test]))