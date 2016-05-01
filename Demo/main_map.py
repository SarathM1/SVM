import numpy as np
import matplotlib.pyplot as plt

"""from sklearn.datasets.base import Bunch
from sklearn import svm, metrics"""

DISEASE_NAME = 'disease_0'


def draw_map(X, Y, land_reference):
    # Plot map of South America"""
    plt.subplot(1, 1, 1)
    print(" - plot coastlines from coverage")
    plt.contour(X, Y, land_reference, levels=[-9999], colors="k",
                linestyles="solid")


def load_from_file(file_name):
    f = file('../'+file_name, "rb")
    data1 = np.load(f)
    data2 = np.load(f)
    data3 = np.load(f)
    data4 = np.load(f)
    data5 = np.load(f)
    data6 = np.load(f)
    data7 = np.load(f)
    data8 = np.load(f)
    f.close()
    return data1, data2, data3, data4, data5, data6, data7, data8


def main():
    xgrid, ygrid, land_reference, coverages_data, test_data, train_data, Nx_data, Ny_data = load_from_file("database.bin")
    X, Y = np.meshgrid(xgrid, ygrid[::-1])

    draw_map(X, Y, land_reference)
    plt.xticks([])
    plt.yticks([])
    plt.title(DISEASE_NAME)
    plt.axis('equal')

    plt.show()
if __name__ == "__main__":
    main()
