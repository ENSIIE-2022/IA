import numpy as np                      # charge un package pour le numérique
import matplotlib.pyplot as plt         # charge un package pour les graphiques 

def printimg(X,y,k):
    X_mean = X[y==k]
    X_mean_mean = np.mean(X_mean, axis =0)
    img = np.reshape(X_mean_mean, (8,8))

    plt.imshow(img, cmap='gray', aspect='equal', interpolation='nearest')
    plt.show()

    pass

for i in classes_list:
    printimg(X,y,i)