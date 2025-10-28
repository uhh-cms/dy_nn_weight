import numpy as np
import matplotlib.pyplot as plt
import math
xpoints = np.random.randint(20)
ypoints = np.linspace(0, 100, 20)

arr = np.zeros((20, 3), dtype=int)
for row in arr:
    row[2] = 4.5

print(arr[1, 3])
plt.plot(xpoints, ypoints, 'go')
plt.plot(ypoints/np.max(ypoints), ypoints, 'bo')
plt.xlabel("xlabel stuff")
plt.ylabel("ylabel thingy")
plt.title("meaningful title")
plt.savefig("my_plot.pdf")
plt.show()

x = np.random.normal(170, 10, 250)
plt.hist(x)
plt.savefig("testhistogram.pdf")
plt.show()

plt.hist(x, 30)
plt.savefig("testhistogram2.pdf")
plt.show()
