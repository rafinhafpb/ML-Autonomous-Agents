list_number = [0]
list = [list(list_number) for _ in range(10)]
print(list)

for i in range(5):
    list[i].append(i)

print(list)