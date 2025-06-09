with open('jing.txt', 'r') as f:
    ans1 = f.readlines()

with open('tsou.txt', 'r') as f:
    ans2 = f.readlines()

print(len(ans1), len(ans2))
print(len(set(ans1) | set(ans2)))

s = set()
ans_final = []

for line in ans1:
    did, label, start, end, text = line.split('\t')
    if (did, start) in s:
        print(line)
        continue
    s.add((did, start))
    ans_final.append(line)

for line in ans2:
    did, label, start, end, text = line.split('\t')
    if (did, start) in s:
        # print(line)
        continue
    s.add((did, start))
    ans_final.append(line)


ans_final.sort(key=lambda line: (int(line.split('\t')[0]), float(line.split('\t')[2])))
# print(len(ans_final))

with open('merge.txt', 'w') as f:
    f.writelines(ans_final)