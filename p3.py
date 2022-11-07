import numpy as np
import matplotlib.pyplot as plt
import h5py
import juliet
import os
import utils as utl
import matplotlib.gridspec as gd

f1 = h5py.File(os.getcwd() + '/Stage4/S4_2022-11-03_wasp39_run2/ap6_bg9/S4_wasp39_ap6_bg9_LCData.h5')
instruments = np.array([])
for i in range(len(f1['wave_mid'])):
    instruments = np.hstack((instruments,'CH' + str(i)))

for i in range(len(instruments)):
    # Making data
    tim7, fl7, fle7, ycen7 = np.asarray(f1['time']), np.asarray(f1['data'][i]), np.asarray(f1['err'][i]), np.asarray(f1['centroid_y'])
    tim7 = tim7 + 2400000.5
    # Removing nans
    tim7, fl7, fle7, ycen7 = tim7[~np.isnan(fl7)], fl7[~np.isnan(fl7)], fle7[~np.isnan(fl7)], ycen7[~np.isnan(fl7)]
    ## Outlier removal: ycen
    msk1 = utl.outlier_removal_ycen(ycen7)
    tim7, fl7, fle7, ycen7 = tim7[msk1], fl7[msk1], fle7[msk1], ycen7[msk1]
    ## Outlier removal: time
    msk2 = utl.outlier_removal(tim7, fl7, fle7, clip=10)
    tim7, fl7, fle7 = tim7[msk2], fl7[msk2], fle7[msk2]
    # Data that juliet understand
    tim, fl, fle = {}, {}, {}