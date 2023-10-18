import numpy as np
from jwst import datamodels
import multiprocessing
multiprocessing.set_start_method('fork')
from jwst.pipeline import calwebb_detector1
from glob import glob
from path import Path
from tqdm import tqdm
import os
import utils
import time

# ------------------------------------------
# For the final analysis: NIRCam SW data
# ------------------------------------------

# This file is to generate corrected data from uncal files:
# Steps: Stage 1 of the JWST pipeline, without dark correction and jump step
# Further steps: correcting errorbars for zeros and Nan and creating a bad-pixel map

segs = []
for i in range(4):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))

p1 = os.getcwd() + '/Data/NRC_SW'
p2 = os.getcwd() + '/RateInts/Corr_NRCSW'    # To store corrected files


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
        fname = glob(p1 + '/*' + seg + '_nrca3_uncal.fits')[0]
        uncal = datamodels.RampModel(fname)
        times_bjd = uncal.int_times['int_mid_BJD_TDB']

        # Stage 1 pipeline
        groupscale_results = calwebb_detector1.group_scale_step.GroupScaleStep.call(uncal, save_results=False)
        dq_results = calwebb_detector1.dq_init_step.DQInitStep.call(groupscale_results, save_results=False)
        saturation_results = calwebb_detector1.saturation_step.SaturationStep.call(dq_results, save_results=False)
        superbias_results = calwebb_detector1.superbias_step.SuperBiasStep.call(saturation_results, save_results=False)

        ## ROEBA algorithm
        m1 = np.ones(superbias_results.data[0,-1,:,:].shape)
        m1[:,1060-150:1060+150] = 0.
        # Computing ROEBA background subtraction on _group_ level
        for integration in tqdm(range(superbias_results.data.shape[0])):
            for group in range(superbias_results.data.shape[1]):
                superbias_results.data[integration, group, :, :] = \
                    utils.roeba_backgroud_sub(superbias_results.data[integration, group, :, :], m1)
        
        linearity_results = calwebb_detector1.linearity_step.LinearityStep.call(superbias_results, save_results=False)
        jumpstep_results = calwebb_detector1.jump_step.JumpStep.call(linearity_results, rejection_threshold=30., maximum_cores='half', save_results=False)
        rampfitting_results = calwebb_detector1.ramp_fit_step.RampFitStep.call(jumpstep_results, maximum_cores='half', save_results=False)
        gainscale_results = calwebb_detector1.gain_scale_step.GainScaleStep.call(rampfitting_results[1], output_dir=os.getcwd() + '/RateInts/Ramp_NRCSW', save_results=True)

        # Data quality mask
        dq = rampfitting_results[1].dq
        mask = np.ones(dq.shape)
        mask[dq > 0] = 0.

        # Time
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
        mask_badpix = mask * mask_bp3 
        ## Mask with cosmic rays
        ### Essentially this mask will have 0s in the places of bad pixels...
        mask_bcr = utils.identify_crays(gainscale_results.data, mask_badpix)
        non_msk_pt_fr = np.sum(mask_bcr) / (mask_bcr.shape[0] * mask_bcr.shape[1] * mask_bcr.shape[2])
        print('---- Total per cent of masked points: {:.4f} %'.format(100 * (1 - non_msk_pt_fr)))
        print('>>>> --- Done!!')

        print('>>>> --- Correcting data...')
        corrected_data = np.copy(gainscale_results.data)
        corrected_data[mask_bcr == 0] = np.nan
        for i in range(corrected_data.shape[0]):
            corrected_data[i,:,:] = utils.replace_nan(corrected_data[i,:,:])
        print('>>>> --- Done!!')

        np.save(p2 + '/Corrected_data_seg' + seg + '.npy', corrected_data[:,:,512:1536])
        np.save(p2 + '/Corrected_errors_seg' + seg + '.npy', corrected_errs[:,:,512:1536])
        np.save(p2 + '/Mask_bcr_seg' + seg + '.npy', mask_bcr[:,:,512:1536])
        np.save(p2 + '/Times_bjd_seg' + seg + '.npy', times_bjd)
    else:
        print('>>>> --- All is already done!!')
        print('>>>> --- Moving on to the next segment!')
    t2 = time.time()
    print('>>>> --- Total time taken: {:.4f} minutes'.format((t2-t1)/60))
    print('>>>> --- --- --- --- --- --- --- --- --- ---')
    print('>>>> ---  Segment ' + str(seg) + ' analysis completed!!')
    print('>>>> --- --- --- --- --- --- --- --- --- ---')