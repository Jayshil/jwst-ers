import numpy as np
import os
from jwst import datamodels
from path import Path
import time
import utils
from tqdm import tqdm
from stark import reduce

# This file is to generate a corrected data, errors and masks for rampfitting file
# (I can do this, because my method to correct data is the same for all analysis: stark and poetss)

visit = 'NIRCam'

segs = []
for i in range(4):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))
#segs = segs[-2:]
p1 = os.getcwd() + '/Data/' + visit
p2 = os.getcwd() + '/RateInts/Corr_' + visit    # To store corrected files
p3 = os.getcwd() + '/RateInts/Ramp_' + visit

input('Did you change the input file names...??')

for seg in range(len(segs)):
    t1 = time.time()
    print('---------------------------------')
    print('Working on Segment: ', segs[seg])
    print('')
    f12 = Path(p2 + '/Corrected_data_seg' + segs[seg] + '.npy')
    f13 = Path(p2 + '/Corrected_errors_seg' + segs[seg] + '.npy')
    f14 = Path(p2 + '/Mask_bcr_seg' + segs[seg] + '.npy')
    f15 = Path(p2 + '/Times_bjd_seg' + segs[seg] + '.npy')
    if f12.exists() and f13.exists() and f14.exists() and f15.exists():
        print('>>>> --- The files already exist...')
        print('         Continuing to the next segment...')
        continue
    else:
        pass
    
    print('>>>> --- Reading dataset...')
    uncal = datamodels.RampModel(p1 + '/' + 'jw01366002001_04103_00001-seg' + segs[seg] + '_nrcalong_uncal.fits')
    times_bjd = uncal.int_times['int_mid_BJD_TDB']
    rampfitting_results = datamodels.DataModel(p3 + '/jw01366002001_04103_00001-seg' + segs[seg] + '_nrcalong_1_rampfitstep.fits')
    darkdq = np.load(p3 + '/bmap_seg' + segs[seg] + '.npy')
    print('>>>> --- Done!!')

    print('>>>> --- Correcting errorbars (for zeros and NaNs)...')
    ## Correct errorbars
    med_err = np.nanmedian(rampfitting_results.err.flatten())
    ## Changing Nan's and zeros in error array with median error
    corr_err1 = np.copy(rampfitting_results.err)
    corr_err2 = np.where(rampfitting_results.err != 0., corr_err1, med_err)                     # Replacing error == 0 with median error
    corrected_errs = np.where(np.isnan(rampfitting_results.err) != True, corr_err2, med_err)    # Replacing error == Nan with median error
    print('>>>> --- Done!!')

    print('>>>> --- Creating a bad-pixel map...')
    ## Making a bad-pixel map
    mask_bp1 = np.ones(rampfitting_results.data.shape)
    mask_bp2 = np.where(rampfitting_results.err != 0., mask_bp1, 0.)                 # This will place 0 in mask where errorbar == 0
    mask_bp3 = np.where(np.isnan(rampfitting_results.err) != True, mask_bp2, 0.)     # This will place 0 in mask where errorbar is Nan
    mask_badpix = np.where(darkdq == 0., mask_bp3, 0.)                               # This will place 0 in mask where darkdq != 0
    ## Mask with cosmic rays
    ### Essentially this mask will have 0s in the places of bad pixels...
    mask_bcr = utils.identify_crays(rampfitting_results.data, mask_badpix)
    print('>>>> --- Done!!')

    if visit == 'NIRCam':
        print('>>>> --- Row-by-row background subtraction...')
        # Row-by-row background subtraction
        bkg_corr_data1 = np.copy(rampfitting_results.data)
        for integrations in tqdm(range(bkg_corr_data1.shape[0])):
            ## Let's first create a mask
            mask = np.ones(rampfitting_results.data[0, :, :].shape)
            for i in range(mask.shape[1]):
                if i<1800:
                    mask[:,i] = np.zeros(len(mask[:,i]))
            mask = mask * mask_bcr[integrations,:,:]
            bkg_corr_data1[integrations, :, :] = reduce.row_by_row_bkg_sub(rampfitting_results.data[integrations,:,:], mask)
        print('>>>> --- Done!!')
    else:
        bkg_corr_data1 = np.copy(rampfitting_results.data)

    print('>>>> --- Column-by-column background subtraction...')
    bkg_corr_data = np.ones(rampfitting_results.data.shape)
    for integrations in tqdm(range(bkg_corr_data.shape[0])):
        # Let's first create a mask!!
        mask = np.ones(rampfitting_results.data[0, :, :].shape)
        if visit == 'NIRSpec':
            for i in range(mask.shape[1]):
                mask[int(7):int(25), int(i)] = 0.
        else:
            for i in range(mask.shape[1]):
                mask[int(0):int(70), int(i)] = 0.
        mask = mask * mask_bcr[integrations,:,:]
        bkg_corr_data[integrations,:,:] =\
            reduce.col_by_col_bkg_sub(bkg_corr_data1[integrations,:,:], mask)
    print('>>>> --- Done!!')

    print('>>>> --- Correcting data...')
    corrected_data = np.copy(bkg_corr_data)
    corrected_data[mask_bcr == 0] = np.nan
    for i in range(corrected_data.shape[0]):
        corrected_data[i,:,:] = utils.replace_nan(corrected_data[i,:,:])
    print('>>>> --- Done!!')

    print('>>> --- Saving results...')
    np.save(p2 + '/Corrected_data_seg' + segs[seg] + '.npy', corrected_data)
    np.save(p2 + '/Corrected_errors_seg' + segs[seg] + '.npy', corrected_errs)
    np.save(p2 + '/Mask_bcr_seg' + segs[seg] + '.npy', mask_bcr)
    np.save(p2 + '/Times_bjd_seg' + segs[seg] + '.npy', times_bjd)
    print('>>>> --- Done!!')

    t2 = time.time()
    print('>>>> --- Total time taken: {:.4f} minutes'.format((t2-t1)/60))