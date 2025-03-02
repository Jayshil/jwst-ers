import numpy as np
from jwst import datamodels
from jwst.pipeline import calwebb_detector1
from glob import glob
from pathlib import Path
from tqdm import tqdm
from stark import reduce
import os
import utils
import time

# This file is to generate a corrected data, errors and masks for rampfitting file
# For NIRSpec Prism data

# Steps: Stage 1 of the JWST pipeline (refpix step and group level background subtraction)
# Further steps: correcting errorbars for zeros and Nan and creating a bad-pixel map

segs = []
for i in range(4):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))

p1 = os.getcwd() + '/Data/NRSPR'
p2 = os.getcwd() + '/RateInts/Corr_NRSPR'    # To store corrected files

for i in range(len(segs)):
    t1 = time.time()
    # Segment no:
    seg = segs[i]
    print('>>>> --- --- --- --- --- --- --- --- --- ---')
    print('>>>> --- Working on Segment ' + str(seg))
    print('>>>> --- --- --- --- --- --- --- --- --- ---')
    # See if outputs are there or not
    f1 = Path(p2 + '/Corrected_data_seg' + seg + '.npy')
    f2 = Path(p2 + '/Corrected_errors_seg' + seg + '.npy')
    f3 = Path(p2 + '/Mask_bcr_seg' + seg + '.npy')
    f4 = Path(p2 + '/Times_bjd_seg' + seg + '.npy')

    if f1.exists() and f2.exists() and f3.exists() and f4.exists():
        all_completed = True
    else:
        all_completed = False
    
    if not all_completed:
        print('>>>> --- Stage 1 processing Starts...')
        fname = glob(p1 + '/*' + seg + '_nrs1_uncal.fits')[0]
        uncal = datamodels.RampModel(fname)
        times_bjd = uncal.int_times['int_mid_BJD_TDB']

        # Stage 1 pipeline
        groupscale_results = calwebb_detector1.group_scale_step.GroupScaleStep.call(uncal, save_results=False)
        dq_results = calwebb_detector1.dq_init_step.DQInitStep.call(groupscale_results, save_results=False)
        saturation_results = calwebb_detector1.saturation_step.SaturationStep.call(dq_results, save_results=False)
        superbias_results = calwebb_detector1.superbias_step.SuperBiasStep.call(saturation_results, save_results=False)
        refpix_results = calwebb_detector1.refpix_step.RefPixStep.call(superbias_results, save_results=False)

        ## ROEBA algorithm
        ### Mask for odd-even effect and slow-read correction
        m1 = np.ones(refpix_results.data[0,-1,:,:].shape)
        m1[8:22,15:475] = 0.

        ## Column-by-column background subtraction on group level
        for integration in tqdm(range(refpix_results.data.shape[0])):
            for group in range(refpix_results.data.shape[1]):
                refpix_results.data[integration, group, :, :], _ =\
                    reduce.col_by_col_bkg_sub(frame=refpix_results.data[integration, group, :, :], mask=m1)
                
        ## Continuing with the rest of the pipeline
        linearity_results = calwebb_detector1.linearity_step.LinearityStep.call(refpix_results, save_results=False)
        darkcurrent_results = calwebb_detector1.dark_current_step.DarkCurrentStep.call(linearity_results, save_results=False)
        rampfitting_results = calwebb_detector1.ramp_fit_step.RampFitStep.call(darkcurrent_results, save_results=False)
        gainscale_results = calwebb_detector1.gain_scale_step.GainScaleStep.call(rampfitting_results[1], output_dir=os.getcwd() + '/RateInts/Ramp_NRSPR', save_results=True)

        # Data quality mask
        dq = rampfitting_results[1].dq
        mask = np.ones(dq.shape)
        mask[dq > 0] = 0.

        ## Time
        times_bjd = uncal.int_times['int_mid_BJD_TDB']

        print('>>>> --- Correcting errorbars (for zeros and NaNs)...')
        ## Correct errorbars
        med_err = np.nanmedian(gainscale_results.err.flatten())
        ## Changing Nan's and zeros in error array with median error
        corr_err1 = np.copy(gainscale_results.err)
        corr_err2 = np.where(gainscale_results.err != 0., corr_err1, med_err)                     # Replacing error == 0 with median error
        corrected_errs = np.where(np.isnan(gainscale_results.err) != True, corr_err2, med_err)    # Replacing error == Nan with median error
        print('>>>> --- Done!!')

        print('>>>> --- Creating a bad-pixel map...')
        ## Making a bad-pixel map
        mask_bp1 = np.ones(gainscale_results.data.shape)
        mask_bp2 = np.where(gainscale_results.err != 0., mask_bp1, 0.)                 # This will place 0 in mask where errorbar == 0
        mask_bp3 = np.where(np.isnan(gainscale_results.err) != True, mask_bp2, 0.)     # This will place 0 in mask where errorbar is Nan
        #mask_badpix = np.where(dq == 0., mask_bp3, 0.)                               # This will place 0 in mask where darkdq != 0
        mask_badpix = mask * mask_bp3                                                 # Adding those pixels which are identified as bad by the pipeline (and hence 0)

        ## Mask with cosmic rays
        ### Essentially this mask will add 0s in the places of bad pixels...
        mask_bcr = utils.identify_crays(gainscale_results.data, mask_badpix)
        non_msk_pt_fr = np.sum(mask_bcr) / (mask_bcr.shape[0] * mask_bcr.shape[1] * mask_bcr.shape[2])
        print('---- Total per cent of masked points: {:.4f} %'.format(100 * (1 - non_msk_pt_fr)))
        print('>>>> --- Done!!')

        print('>>>> --- Correcting data...')
        corrected_data = np.copy(gainscale_results.data)
        corrected_data[mask_bcr == 0] = np.nan
        for integration in range(corrected_data.shape[0]):
            corrected_data[integration,:,:] = utils.replace_nan(corrected_data[integration,:,:])
        print('>>>> --- Done!!')

        print('>>>> --- Additional background correction...')
        corrected_data_bkg = np.ones(corrected_data.shape)
        for integration in tqdm(range(corrected_data.shape[0])):
            corrected_data_bkg[integration,:,:], _ = reduce.col_by_col_bkg_sub(corrected_data[integration,:,:], mask=m1*mask_bcr[i,:,:])
        print('>>>> --- Done!!')

        np.save(p2 + '/Corrected_data_seg' + seg + '.npy', corrected_data_bkg)
        np.save(p2 + '/Corrected_errors_seg' + seg + '.npy', corrected_errs)
        np.save(p2 + '/Mask_bcr_seg' + seg + '.npy', mask_bcr)
        np.save(p2 + '/Times_bjd_seg' + seg + '.npy', times_bjd)
    else:
        print('>>>> --- All is already done!!')
        print('>>>> --- Moving on to the next segment!')
    t2 = time.time()
    print('>>>> --- Total time taken: {:.4f} minutes'.format((t2-t1)/60))
    print('>>>> --- --- --- --- --- --- --- --- --- ---')
    print('>>>> ---  Segment ' + str(seg) + ' analysis completed!!')
    print('>>>> --- --- --- --- --- --- --- --- --- ---')