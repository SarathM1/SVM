from sklearn.datasets import fetch_species_distributions
from sklearn.datasets.species_distributions import construct_grids
import numpy as np


def save_to_file(file_name, data1, data2, data3, data4, data5, data6, data7, data8):
    f = file(file_name, "wb")
    np.save(f, data1)
    np.save(f, data2)
    np.save(f, data3)
    np.save(f, data4)
    np.save(f, data5)
    np.save(f, data6)
    np.save(f, data7)
    np.save(f, data8)
    f.close()


def load_from_file(file_name):
    f = file(file_name, "rb")
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


def replace(data):
    x = [ 'bradypus_variegatus_0',
            'microryzomys_minutus_0',
            'bradypus_variegatus_1',
            'microryzomys_minutus_1',
            'bradypus_variegatus_2',
            'microryzomys_minutus_2',
            'bradypus_variegatus_3',
            'microryzomys_minutus_3',
            'bradypus_variegatus_4',
            'microryzomys_minutus_4',
            'bradypus_variegatus_5',
            'microryzomys_minutus_5',
            'bradypus_variegatus_6',
            'microryzomys_minutus_6',
            'bradypus_variegatus_7',
            'microryzomys_minutus_7',
            'bradypus_variegatus_8',
            'microryzomys_minutus_8',
            'bradypus_variegatus_9',
            'microryzomys_minutus_9',
            ]
   
    """
    print '[',
    for each in x:
        print "'"+each+"'"+','
    print ']'
    """

    for i in range(20):
        cnt = 0
        print x[i]
        for j in range(len(data)):
            if data[j][0] == x[i]:
                print data[j][0], '=>', 'disease_' + str(i)
                cnt+=1
                data[j][0] = 'disease_' + str(i)
                flag = j
        # print "Count = ",cnt
        print data[flag][0], "Count = ", cnt

# Extracting Data from web
data = fetch_species_distributions()

# Print all the subsection available
print "keys = ", data.keys()

print "Data = "
print "_"*80
xgrid, ygrid = construct_grids(data)
replace(data.test)
replace(data.train)
save_to_file("database.bin", xgrid, ygrid, data.coverages[6], data.coverages, data.test, data.train, data.Nx, data.Ny)
print len(xgrid), len(ygrid), len(data.coverages[6]), len(data.coverages), len(data.test), len(data.train), data.Nx, data.Ny
print "_"*80

a, b, c, d, e, f, g, h = load_from_file("database.bin")
print len(a), len(b), len(c), len(d), len(e), len(f), g, h
print "_"*80
