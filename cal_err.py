import numpy as np

f = open("polygon_18.txt", "r")
polygon = []
for i in range(20):
    line = float(f.readline().split()[0])
    polygon.append(line)
f.close()
print(polygon)

print("mean:", np.mean(polygon))
print("std:", np.std(polygon))

f = open("result.txt", "a")
f.write("mean: {}\n".format(np.mean(polygon)))
f.write("std: {}".format(np.std(polygon)))
f.close()