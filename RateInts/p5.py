import numpy as np
from jwst import datamodels
from poetss import poetss
from stark import reduce
from pathlib import Path
from glob import glob
from tqdm import tqdm
import os
import utils
import time

# This file is to generate a corrected data, errors and masks for rampfitting file
# For MIRI data

# Steps: we take the rateints data from the default JWST pipeline
# Further steps: correcting errorbars for zeros and Nan and creating a bad-pixel map

segs = []
for i in range(10):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))

detector = 'mirimage'

p1 = os.getcwd() + '/RateInts/Ramp_MIRI'
p2 = os.getcwd() + '/RateInts/Corr_MIRI'    # To store corrected files

for se in range(len(segs)):
    t1 = time.time()
    # Segment no:
    seg = segs[se]
    fragments = ['00001', '00002', '00003']

    print('>>>> --- --- --- --- --- --- --- --- --- --- ---')
    print('>>>> --- Working on Segment ' + str(seg))
    print('>>>> --- --- --- --- --- --- --- --- --- --- ---')

    for fr in range(len(fragments)):
        print('>>>> --- Fragment: ', fragments[fr])
        # See if outputs are there or not
        f1 = Path(p2 + '/Corrected_data_' + detector + '_' + fragments[fr] + '_seg' + seg + '.npy')
        f2 = Path(p2 + '/Corrected_errors_' + detector + '_' + fragments[fr] + '_seg' + seg + '.npy')
        f3 = Path(p2 + '/Mask_bcr_' + detector + '_' + fragments[fr] + '_seg' + seg + '.npy')
        f4 = Path(p2 + '/Times_bjd_' + detector + '_' + fragments[fr] + '_seg' + seg + '.npy')

        if f1.exists() and f2.exists() and f3.exists() and f4.exists():
            all_completed = True
        else:
            all_completed = False
        
        if not all_completed:
            print('>>>> --- Stage 2 processing Starts...')
            fname = glob(p1 + '/*' + fragments[fr] + '-seg' + seg + '_' + detector + '_rateints.fits')[0]
            rate_ints = datamodels.open(fname)
            times_bjd = rate_ints.int_times['int_mid_BJD_TDB']

            ## Bad-pixel map
            dq = rate_ints.dq
            mask = np.ones(dq.shape)
            mask[dq > 0] = 0.

            print('>>>> --- Correcting errorbars (for zeros and NaNs)...')
            ## Correct errorbars
            med_err = np.nanmedian(rate_ints.err.flatten())
            ## Changing Nan's and zeros in error array with median error
            corr_err1 = np.copy(rate_ints.err)
            corr_err2 = np.where(rate_ints.err != 0., corr_err1, med_err)                     # Replacing error == 0 with median error
            corrected_errs = np.where(np.isnan(rate_ints.err) != True, corr_err2, med_err)    # Replacing error == Nan with median error
            print('>>>> --- Done!!')

            print('>>>> --- Creating a bad-pixel map...')
            ## Making a bad-pixel map
            mask_bp1 = np.ones(rate_ints.data.shape)
            mask_bp2 = np.where(rate_ints.err != 0., mask_bp1, 0.)                 # This will place 0 in mask where errorbar == 0
            mask_bp3 = np.where(np.isnan(rate_ints.err) != True, mask_bp2, 0.)     # This will place 0 in mask where errorbar is Nan
            #mask_badpix = np.where(dq == 0., mask_bp3, 0.)                               # This will place 0 in mask where darkdq != 0
            mask_badpix = mask * mask_bp3                                                 # Adding those pixels which are identified as bad by the pipeline (and hence 0)

            ## Mask with cosmic rays
            ### Essentially this mask will add 0s in the places of bad pixels...
            mask_bcr = utils.identify_crays(rate_ints.data, mask_badpix)
            print('>>>> --- Done!!')

            print('>>>> --- Correcting data...')
            corrected_data = np.copy(rate_ints.data)
            corrected_data[mask_bcr == 0] = np.nan
            for i in range(corrected_data.shape[0]):
                corrected_data[i,:,:] = utils.replace_nan(corrected_data[i,:,:])
            print('>>>> --- Done!!')

            print('>>>> --- Finding trace positions in order to perform background subtraction...')
            # Finding trace
            xstart, xend = 143, 394
            try:
                cent_cub1 = poetss.find_trace_cof(clean_cube=np.transpose(corrected_data[:,xstart:xend,:], axes=[0,2,1]), margin=5)
                trace1, dx1 = poetss.fit_multi_trace(cent_mat=cent_cub1, deg=3, clip=3)
            except:
                trace1 = np.load(os.getcwd() + '/MIRI/Outputs/Tests/traces_miri.npy')
            xpos = np.arange(xstart, xend, 1)
            print('>>>> --- Done!!')

            print('>>>> --- Performing background subtraction...')
            mask_bkg = np.ones(corrected_data[0,:,:].shape)
            mask_bkg[:,0:10] = 0.
            mask_bkg[:,62:] = 0.
            for i in range(len(xpos)):
                mask_bkg[int(xpos[i]), int(trace1[i]-7):int(trace1[i]+7+1)] = 0.
            
            corrected_data_bkg = np.ones(corrected_data.shape)
            for i in tqdm(range(corrected_data.shape[0])):
                # For median background subtraction
                int_bkg_corr_data, _ = reduce.col_by_col_bkg_sub(corrected_data[i,:,:].T,\
                                                                        mask=mask_bkg.T*mask_bcr[i,:,:].T)
                corrected_data_bkg[i,:,:] = np.transpose( int_bkg_corr_data )

            print('>>>> --- Done!!')

            np.save(p2 + '/Corrected_data_' + detector + '_' + fragments[fr] + '_seg' + seg + '.npy', corrected_data_bkg)
            np.save(p2 + '/Corrected_errors_' + detector + '_' + fragments[fr] + '_seg' + seg + '.npy', corrected_errs)
            np.save(p2 + '/Mask_bcr_' + detector + '_' + fragments[fr] + '_seg' + seg + '.npy', mask_bcr)
            np.save(p2 + '/Times_bjd_' + detector + '_' + fragments[fr] + '_seg' + seg + '.npy', times_bjd)
        else:
            print('>>>> --- All is already done!!')
            print('>>>> --- Moving on to the next segment!')
    t2 = time.time()
    print('>>>> --- Total time taken: {:.4f} minutes'.format((t2-t1)/60))
    print('>>>> --- --- --- --- --- --- --- --- --- ---')
    print('>>>> ---  Segment ' + str(seg) + ' analysis completed!!')
    print('>>>> --- --- --- --- --- --- --- --- --- ---')