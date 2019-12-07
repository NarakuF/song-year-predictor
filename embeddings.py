"""
Author:
Zhehan Shi      - zs2442@columbia.edu
Guandong Liu    - gl2675@columbia.edu
Yue Wan         - yw3373@columbia.edu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

citys = ['BOS', 'NYC', 'DC', 'MIA', 'CHI', 'SEA', 'SF', 'LA', 'DEN']
distance = np.array([
    [0, 206, 429, 1504, 963, 2976, 3095, 2979, 1949],
    [206, 0, 233, 1308, 802, 2815, 2934, 2786, 1771],
    [429, 233, 0, 1075, 671, 2684, 2799, 2631, 1616],
    [1504, 1308, 1075, 0, 1329, 3273, 3053, 2687, 2037],
    [963, 802, 671, 1329, 0, 2013, 2142, 2054, 996],
    [2976, 2815, 2684, 3273, 2013, 0, 808, 1131, 1307],
    [3095, 2934, 2799, 3053, 2142, 808, 0, 379, 1235],
    [2979, 2786, 2631, 2687, 2054, 1131, 379, 0, 1059],
    [1949, 1771, 1616, 2037, 996, 1307, 1235, 1059, 0]
])

pca = PCA(n_components=2, svd_solver='full')
pca.fit(distance)

result = pca.components_

fig, ax = plt.subplots()
ax.scatter(result[0], result[1])
for i in range(len(citys)):
    ax.annotate(citys[i], (result[0][i], result[1][i]))
# plt.savefig('city_map.png', dpi=512)
