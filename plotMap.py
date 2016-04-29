from time import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_species_distributions
from sklearn.datasets.species_distributions import construct_grids
from sklearn.datasets.base import Bunch
from sklearn import svm, metrics

"""from sklearn.datasets.base import Bunch
from sklearn import svm, metrics"""

species=("bradypus_variegatus_0", "microryzomys_minutus_0")
def create_species_bunch(species_name, train, test, coverages, xgrid, ygrid):
    """Create a bunch with information about a particular organism

    This will use the test/train record arrays to extract the
    data specific to the given species name.
    """
    bunch = Bunch(name=' '.join(species_name.split("_")[:2]))
    species_name = species_name.encode('ascii')
    points = dict(test=test, train=train)

    for label, pts in points.items():
        # choose points associated with the desired species
        pts = pts[pts['species'] == species_name]
        bunch['pts_%s' % label] = pts

        # determine coverage values for each of the training & testing points
        ix = np.searchsorted(xgrid, pts['dd long'])
        iy = np.searchsorted(ygrid, pts['dd lat'])
        bunch['cov_%s' % label] = coverages[:, -iy, ix].T

    return bunch

def draw_map(data, xgrid, ygrid, X, Y, land_reference):
    # Plot map of South America"""
    plt.subplot(1, 1, 1)
    print(" - plot coastlines from coverage")
    plt.contour(X, Y, land_reference,
            levels=[-9999], colors="k",
            linestyles="solid")

def SVM_fun(data, land_reference, mean, std):
    # Fit OneClassSVM
    train_cover_std = (species.cov_train - mean) / std
    print " - fit OneClassSVM ... "
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.5)
    clf.fit(train_cover_std)
    print "done"

    print(" - predict species distribution")

    # Predict species distribution using the training data

    # We'll predict only for the land points.
    idx = np.where(land_reference > -9999)
    coverages_land = data.coverages[:, idx[0], idx[1]].T

    pred = clf.decision_function((coverages_land - mean) / std)[:, 0]
    
    return pred, clf, idx

def compute_AUC(Z, data, clf, mean, std):
    # Compute AUC with regards to background points
    np.random.seed(13)
    background_points = np.c_[np.random.randint(low=0, high=data.Ny,
                                                size=10000),
                              np.random.randint(low=0, high=data.Nx,
                                                size=10000)].T
    pred_background = Z[background_points[0], background_points[1]]
    pred_test = clf.decision_function((species.cov_test - mean)
                                        / std)[:, 0]
    scores = np.r_[pred_test, pred_background]
    y = np.r_[np.ones(pred_test.shape), np.zeros(pred_background.shape)]
    fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    roc_auc = metrics.auc(fpr, tpr)
    plt.text(-35, -70, "AUC: %.3f" % roc_auc, ha="right")
    print("\n Area under the ROC curve : %f" % roc_auc)

def mark_Prediction(X, Y, Z, levels):
    # plot contours of the prediction
    plt.contourf(X, Y, Z, levels=levels, cmap=plt.cm.Reds)
    plt.colorbar(format='%.2f')

def mark_Points(species):
    # scatter training/testing points
    plt.scatter(species.pts_train['dd long'], species.pts_train['dd lat'],
                s=2 ** 2, c='black',
                marker='^', label='train')
    plt.scatter(species.pts_test['dd long'], species.pts_test['dd lat'],
                s=2 ** 2, c='black',
                marker='x', label='test')
def main():
    global species
    # Extracting Data from web
    data = fetch_species_distributions()
    xgrid, ygrid = construct_grids(data)
    X, Y = np.meshgrid(xgrid, ygrid[::-1])
    land_reference = data.coverages[6]
    BV_bunch = create_species_bunch(species[0],
                                    data.train, data.test,
                                    data.coverages, xgrid, ygrid)

    draw_map(data, xgrid, ygrid, X, Y, land_reference)
    plt.xticks([])
    plt.yticks([])

    for i, species in enumerate([BV_bunch]):
        # Standardize features
        mean = species.cov_train.mean(axis=0)
        std = species.cov_train.std(axis=0)
        
        pred, clf, idx = SVM_fun(data, land_reference, mean, std)
  
        Z = np.ones((data.Ny, data.Nx), dtype=np.float64)
        Z *= pred.min()
        Z[idx[0], idx[1]] = pred

        levels = np.linspace(Z.min(), Z.max(), 25)
        Z[land_reference == -9999] = -9999

        mark_Prediction(X, Y, Z, levels)

        mark_Points(species)
        
        plt.legend()
        plt.title('Title')
        plt.axis('equal')
        
        compute_AUC(Z, data, clf, mean, std)
    plt.show()
if __name__ == "__main__":
    main()

