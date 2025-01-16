p = {}

p['a a a'] = 1
p['b b b'] = 2
p['c x c'] = 3

for keys in p.keys():
    vector = keys.split()[-1]

print(vector)