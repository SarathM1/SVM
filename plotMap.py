from time import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets.base import Bunch
from sklearn.datasets import fetch_species_distributions
from sklearn.datasets.species_distributions import construct_grids
from sklearn import svm, metrics

species=("bradypus_variegatus_0", "microryzomys_minutus_0")
# Extracting Data from web
data = fetch_species_distributions()

# Print all the subsection available

print "keys = ", data.keys()
print "\n\n Hit Enter to Continue"
raw_input()

print "_"*80
print "Data = "
print data['coverages'][6]
print "_"*80


land_reference = data.coverages[6]

# Set up the data grid
xgrid, ygrid = construct_grids(data)

# The grid in x,y coordinates
X, Y = np.meshgrid(xgrid, ygrid[::-1])
# Plot map of South America"""
# for i, species in enumerate([BV_bunch, MM_bunch]):
plt.subplot(1, 2, 1)
print(" - plot coastlines from coverage")
plt.contour(X, Y, land_reference,
        levels=[-9999], colors="k",
        linestyles="solid")
plt.xticks([])
plt.yticks([])
plt.show()
