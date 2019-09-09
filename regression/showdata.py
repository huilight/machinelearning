import os
import numpy as np
from matplotlib import pyplot as plt

xaxis = []
yaxis = []


filename = os.getcwd()+"\prices.txt"
with open(filename, 'r') as f:
    for i in f:
        dat = i.split(',')
        if len(dat) == 2:
            xaxis.append(int(dat[0]))
            yaxis.append(int(dat[1]))

# x = xaxis
# y = yaxis
x = np.array(xaxis)
y = np.array(yaxis)
plt.plot(x,y, "ob")
# plt.plot([1,5,3,7,9], [1,9,3,8,7],"ob")
plt.xlabel('x')
plt.ylabel('y')


# plt.plot(y, x)
plt.show()