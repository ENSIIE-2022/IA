code = {'e':'a', 'l':'m', 'o':'e'}
s = 'Hello world!'
# s_code = 'Hamme wermd!'

s = list(s)

counter = 0
for letter in s:
    if letter in code:
        s[counter] = code[letter]
    counter += 1

print(''.join(s))