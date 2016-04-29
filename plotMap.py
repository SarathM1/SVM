from time import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_species_distributions
from sklearn.datasets.species_distributions import construct_grids

"""from sklearn.datasets.base import Bunch
from sklearn import svm, metrics"""

# Extracting Data from web
data = fetch_species_distributions()

land_reference = data.coverages[6]

# Set up the data grid
xgrid, ygrid = construct_grids(data)

# The grid in x,y coordinates
X, Y = np.meshgrid(xgrid, ygrid[::-1])
# Plot map of South America"""
# for i, species in enumerate([BV_bunch, MM_bunch]):
plt.subplot(1, 1, 1)
print(" - plot coastlines from coverage")
plt.contour(X, Y, land_reference,
        levels=[-9999], colors="k",
        linestyles="solid")
plt.show()
