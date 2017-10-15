import matplotlib.pyplot as plt
import numpy as np
#matplotlib inline
#config InlineBackend.figure_format = 'retina'

def odds(p):
    return p / (1 - p)

x = np.arange(0, 1, 0.05)
odds_x = odds(x)

plt.plot(x, odds_x)
plt.axvline(0.0, color='k')
plt.ylim(-0.1, 15)
plt.xlabel('x')
plt.ylabel('odds(x)')

# y axis ticks and gridline
plt.yticks([0.0, 5, 10])
ax = plt.gca()
ax.yaxis.grid(True)

plt.tight_layout()
plt.show()
