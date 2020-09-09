import numpy as np                      # charge un package pour le numérique
import matplotlib.pyplot as plt         # charge un package pour les graphiques 


mean_class = int(np.mean(y))

for line in range(y.size):
    if y[line] == mean_class:
        plt.imshow(np.reshape(X[line, :], (8, 8)))
        plt.show()
    cpt += 1