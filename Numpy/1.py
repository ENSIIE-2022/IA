import numpy as np

i = 1
n = 10000000

M = np.arange(i,n,1)

M = (4 * (M**2)) / (4 * (M**2) -1 ) 

print(M)
print(M.prod()*2)