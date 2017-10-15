import matplotlib.pyplot as plt
import numpy as np

def log_odds(p):
    return np.log(p / (1 - p))

x = np.arange(0.005, 1, 0.005)
log_odds_x = log_odds(x)

plt.plot(x, log_odds_x)
plt.axvline(0.0, color='k')
plt.ylim(-8, 8)
plt.xlabel('x')
plt.ylabel('log_odds(x)')

# y axis ticks and gridline
plt.yticks([-7, 0, 7])
ax = plt.gca()
ax.yaxis.grid(True)

plt.tight_layout()
plt.show()
