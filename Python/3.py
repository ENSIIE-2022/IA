pi=1
for i in range (1, 1000000):
    pi *= 4 * i * i / (4 *i * i - 1)
pi *= 2
print(pi)