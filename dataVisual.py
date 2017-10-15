# Author: NMHai
# Reference:
# Implement some methods for plotting data.

import pandas as pd;
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import random

data = pd.read_csv("training_data.csv");

depth = data['Depth'];
gr = data['GR'];

#depth = np.random.rand(100000);
#scale = lambda x: x*x +10;
#gr = scale(depth);

fig = plt.figure();
fig.suptitle("Plot", fontsize=20);
plt.xlabel('Depth', fontsize=15);
plt.ylabel('GR', fontsize=15)

plt.scatter(depth, gr);
out_png = 'out_file.png';
plt.savefig(out_png);

# 2D Plot
# colors = ['b', 'c', 'y', 'm', 'r']
#
# lo = plt.scatter(random(10), random(10), marker='x', color=colors[0])
# ll = plt.scatter(random(10), random(10), marker='o', color=colors[0])
# l  = plt.scatter(random(10), random(10), marker='o', color=colors[1])
# a  = plt.scatter(random(10), random(10), marker='o', color=colors[2])
# h  = plt.scatter(random(10), random(10), marker='o', color=colors[3])
# hh = plt.scatter(random(10), random(10), marker='o', color=colors[4])
# ho = plt.scatter(random(10), random(10), marker='x', color=colors[4])
#
# plt.legend((lo, ll, l, a, h, hh, ho),
#            ('Low Outlier', 'LoLo', 'Lo', 'Average', 'Hi', 'HiHi', 'High Outlier'),
#            scatterpoints=1,
#            loc='lower left',
#            ncol=3,
#            fontsize=8)
#
# plt.show()

# 3D Plot
# colors=['b', 'c', 'y', 'm', 'r']
#
# ax = plt.subplot(111, projection='3d')
#
# ax.plot(random(10), random(10), random(10), 'x', color=colors[0], label='Low Outlier')
# ax.plot(random(10), random(10), random(10), 'o', color=colors[0], label='LoLo')
# ax.plot(random(10), random(10), random(10), 'o', color=colors[1], label='Lo')
# ax.plot(random(10), random(10), random(10), 'o', color=colors[2], label='Average')
# ax.plot(random(10), random(10), random(10), 'o', color=colors[3], label='Hi')
# ax.plot(random(10), random(10), random(10), 'o', color=colors[4], label='HiHi')
# ax.plot(random(10), random(10), random(10), 'x', color=colors[4], label='High Outlier')
#
# plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
#
# plt.show()
