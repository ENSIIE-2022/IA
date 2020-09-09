s = "HelLo WorLd!!"
dico ={}

for letter in s:
    if letter in dico:
        dico[letter] += 1
    else:
        dico[letter] = 1

print(dico)