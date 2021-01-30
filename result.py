import numpy as np
from matplotlib import pyplot as plt

x = np.array([6, 12, 18])
y = np.array([8.563663923740387, 8.837507808208466, 9.283443188667297])

yerr = np.array([0.05131095252256465, 0.0384716044660047, 0.3135843575924129])

plt.errorbar(
    x,
    y,
    yerr,
    # fmt="ro-",
    # elinewidth="0.5",
    # ecolor="k",
    # capsize="3",
    # capthick="0.5",
)
plt.show()