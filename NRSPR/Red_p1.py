import numpy as np
import matplotlib.pyplot as plt
from jwst.pipeline import calwebb_detector1
from stark import SingleOrderPSF, optimal_extract
from stark import aperture_extract
from krithika import plotstyles
from krithika.utils import pipe_mad
from jwst import datamodels
from poetss import poetss
from stark import reduce
from tqdm import tqdm
import time
import os
import utils

from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'
rcParams['legend.frameon'] = True
rcParams['figure.dpi'] = 300

# This file is to calibrate the uncal data and produce the rateints file using the JWST data reduction pipeline.
# We will then use the rateints file for the rest of the reduction and analysis.

# -----------------------------------------------------
#   Data path and file name
# -----------------------------------------------------
pin = os.getcwd() + '/Data/NRSPR'
pout1 = os.getcwd() + '/NRSPR/RateInts'
pout2 = os.getcwd() + '/NRSPR/Outputs'

# -----------------------------------------------------
# Aperture radii and sigma clipping threshold
# -----------------------------------------------------
aprad_psf = 7
aprad_extraction = 5
clip_sigma = 20
oversample_psf = 2
xstart, xend = 150, 450

# -----------------------------------------------------
#  List of segments
# -----------------------------------------------------
segments = ['seg' + str(i).zfill(3) for i in range(1, 5)]


for seg in segments:
    t1 = time.time()
    fname = 'jw01366004001_04101_00001-' + seg + '_nrs1_uncal.fits'
    fname_rateints = '_'.join(fname.split('_')[:-1]) + '_gainscalestep.fits'

    # -----------------------------------------------------
    #   Load data and metadata
    # -----------------------------------------------------
    uncal = datamodels.RampModel(pin + '/' + fname)
    nint = np.random.randint(0, 100)#uncal.data.shape[0])

    ## Time
    times_bjd = uncal.int_times['int_mid_BJD_TDB']

    # -----------------------------------------------------
    #    Starting Stage 1 of the JWST pipeline
    # -----------------------------------------------------
    
    if not os.path.isfile(pout1 + '/' + fname_rateints):
        print('Calibrating ' + fname + '...')
        #det1 = calwebb_detector1.Detector1Pipeline.call(uncal,\
        #                                                steps={'jump' : {'rejection_threshold' : 15, 'maximum_cores' : '1'},\
        #                                                       'dark_current' : {'skip' : True},
        #                                                       'ramp_fit' : {'maximum_cores' : '1'}},\
        #                                                output_dir=pout1, save_results=True)

        group_scale_results = calwebb_detector1.group_scale_step.GroupScaleStep.call(uncal, save_results=False)
        dq_results = calwebb_detector1.dq_init_step.DQInitStep.call(group_scale_results, save_results=False)
        saturation_results = calwebb_detector1.saturation_step.SaturationStep.call(dq_results, save_results=False)

        superbias_results = calwebb_detector1.superbias_step.SuperBiasStep.call(saturation_results, save_results=False)
        ## Doing supervias step manually
        """superbias = np.nanmedian(saturation_results.data[:,0,:,:], axis=0)
        for inte in range(saturation_results.data.shape[0]):
            for gr in range(saturation_results.data.shape[1]):
                saturation_results.data[inte,gr,:,:] = saturation_results.data[inte,gr,:,:] - superbias"""

        refpix_results = calwebb_detector1.refpix_step.RefPixStep.call(superbias_results, save_results=False)
        linearity_results = calwebb_detector1.linearity_step.LinearityStep.call(refpix_results, save_results=False)

        ## Manualy background subtraction at group level
        mask_bkg = np.ones(linearity_results.data[0,0,:,:].shape)
        mask_bkg[4:28,:] = 0.
        ### Bad-pixel map
        dq = linearity_results.groupdq
        mask_bp = np.ones(dq.shape)
        mask_bp[dq > 0] = 0.
        for inte in tqdm(range(linearity_results.data.shape[0])):
            for gr in range(linearity_results.data.shape[1]):
                linearity_results.data[inte,gr,:,:], _ = reduce.col_by_col_bkg_sub(frame=linearity_results.data[inte,gr,:,:], mask=mask_bkg * mask_bp[inte,gr,:,:])

        rampfitting_results = calwebb_detector1.ramp_fit_step.RampFitStep.call(linearity_results, save_results=False)
        gainscale_results = calwebb_detector1.gain_scale_step.GainScaleStep.call(rampfitting_results[1], output_dir=pout1, save_results=True)

        # ------ Loading the rateints file
        #rate_ints = datamodels.open(pout + '/' + fname.replace('uncal', 'rateints'))
        #rate_ints = datamodels.open(pout + '/' + fname.replace('uncal', '1_rampfitstep'))
        #rate_ints = datamodels.open(pout + '/' + fname.replace('uncal', 'gainscalestep'))
        
    else:
        print(fname_rateints + ' already exists. Skipping calibration. Loading the rateints file...')
    
    rate_ints = datamodels.open(pout1 + '/' + fname_rateints)

    ## Bad-pixel map
    dq = rate_ints.dq
    mask = np.ones(dq.shape)
    mask[dq > 0] = 0.

    # -------------------------------------------------------------
    #  Correcting errorbars (for zeros and NaNs)
    # -------------------------------------------------------------
    print('>>>> --- Correcting errorbars (for zeros and NaNs)...')
    ## Correct errorbars
    med_err = np.nanmedian(rate_ints.err.flatten())
    ## Changing Nan's and zeros in error array with median error
    corr_err1 = np.copy(rate_ints.err)
    corr_err2 = np.where(rate_ints.err != 0., corr_err1, med_err)                     # Replacing error == 0 with median error
    corrected_errs = np.where(np.isnan(rate_ints.err) != True, corr_err2, med_err)    # Replacing error == Nan with median error
    print('>>>> --- Done!!')

    # And now, adding this "bad" errors to the bad-pixel map
    print('>>>> --- Creating a bad-pixel map...')
    ## Making a bad-pixel map
    mask_bp1 = np.ones(rate_ints.data.shape)
    mask_bp2 = np.where(rate_ints.err != 0., mask_bp1, 0.)                 # This will place 0 in mask where errorbar == 0
    mask_bp3 = np.where(np.isnan(rate_ints.err) != True, mask_bp2, 0.)     # This will place 0 in mask where errorbar is Nan
    #mask_badpix = np.where(dq == 0., mask_bp3, 0.)                               # This will place 0 in mask where darkdq != 0
    mask_badpix = mask * mask_bp3  
    print('>>>> --- Done!!')

    # --------------------------------------------------------------
    #  Identifying the cosmic rays
    # --------------------------------------------------------------
    print('>>>> --- Identifying the cosmic rays...')
    ## Mask with cosmic rays
    ### Essentially this mask will add 0s in the places of bad pixels...
    mask_bcr = utils.identify_crays(rate_ints.data, mask_badpix)
    print('>>>> --- Done!!')

    print('Total per cent of masked points: {:.4f} %'.format(100 * (1 - np.sum(mask_bcr) / (mask_bcr.shape[0] * mask_bcr.shape[1] * mask_bcr.shape[2]))))


    # --------------------------------------------------------------
    #  Correcting the data
    # --------------------------------------------------------------
    print('>>>> --- Correcting data...')
    corrected_data = np.copy(rate_ints.data)
    corrected_data[mask_bcr == 0] = np.nan
    for i in range(corrected_data.shape[0]):
        corrected_data[i,:,:] = utils.replace_nan(corrected_data[i,:,:])
    print('>>>> --- Done!!')


    # --------------------------------------------------------------
    #  Finding the spectral trace
    # --------------------------------------------------------------
    # Finding trace
    cent_cub1 = poetss.find_trace_cof(clean_cube=corrected_data[:,:,xstart:xend], margin=5)
    trace1, dx1 = poetss.fit_multi_trace(cent_mat=cent_cub1, deg=3, clip=3)
    xpos = np.arange(xstart, xend, 1)

    # --------------------------------------------------------------
    #    Background correction
    # --------------------------------------------------------------
    ## Background mask
    mask_bkg = np.ones(corrected_data[nint,:,:].shape)
    for i in range(len(xpos)):
        ystart, yend = int(trace1[i]-10), int(trace1[i]+10+1)
        ## Putting limits to avoid going out of bounds
        if ystart < 0:
            ystart = 0
        if yend > corrected_data.shape[1]:
            yend = corrected_data.shape[1]
        mask_bkg[ystart:yend, int(xpos[i])] = 0.
    
    for i in tqdm(range(corrected_data.shape[0])):
        #if detector == 'nrs1':
        #    corrected_data[i,:,:], _ = reduce.col_by_col_bkg_sub(corrected_data[i,:,:], mask=mask_bkg*mask_bcr[i,:,:])
        #else:
        corrected_data[i,:,:], _ = reduce.col_by_col_bkg_sub(corrected_data[i,:,:], mask=mask_bkg*mask_badpix[i,:,:])

    # --------------------------------------------------------------
    #  Fitting 1D spline to the PSF
    # --------------------------------------------------------------
    ## Identifying bad integrations
    bad_ints, _, _ = np.where(np.isnan(corrected_data[:,:,xstart:xend]))
    bad_ints = np.unique(bad_ints)

    ## Creating a mask for bad integrations
    mask_bad_ints = np.ones(corrected_data.shape[0], dtype=bool)
    mask_bad_ints[bad_ints] = False

    ## Masking bad integrations
    corrected_data, corrected_errs = corrected_data[mask_bad_ints, :, :], corrected_errs[mask_bad_ints, :, :]
    mask_bcr = mask_bcr[mask_bad_ints, :, :]
    times_bjd = times_bjd[mask_bad_ints]

    # Finding trace
    cent_cub1 = poetss.find_trace_cof(clean_cube=corrected_data[:,:,xstart:xend], margin=5)
    trace1, dx1 = poetss.fit_multi_trace(cent_mat=cent_cub1, deg=3, clip=3)
    xpos = np.arange(xstart, xend, 1)

    traces = {}
    traces['xpos'], traces['trace1'], traces['dx1'] = xpos, trace1, dx1

    ## Saving the trace and the x positions for later use
    np.savez(pout2 + '/Trace_' + seg + '.npz', xpos=xpos, traces=trace1, dx1=dx1)

    plt.figure(figsize=(15,5))
    im = plt.imshow(corrected_data[nint,:,:], interpolation='none', aspect='auto')
    im.set_clim([0,1e2])
    plt.plot(xpos, trace1, 'k-')
    plt.title('Example data with the location of spectral trace')
    plt.savefig(pout2 + '/Figs/Trace_loc_' + seg + '.png', bbox_inches='tight')
    plt.close()

    # Plotting the trace position as a function of time
    plt.figure(figsize=(15,5))
    plt.plot( (times_bjd - times_bjd[0])*24, dx1, 'k-', lw=1.)
    plt.xlabel('Time since beginning [hr]')
    plt.ylabel('Position of spectral trace [pix]')

    plt.xlim(0, (times_bjd[-1] - times_bjd[0])*24)
    plt.savefig(pout2 + '/Figs/Trace_jitter_' + seg + '.png', bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------
    #  Converting 1D trace position to 2D
    #  (i.e., trace for every frame)
    # ---------------------------------------------------
    ypos2d = np.zeros((corrected_data.shape[0], len(xpos)))
    for i in range(ypos2d.shape[0]):
        ypos2d[i,:] = trace1 + dx1[i]

    # ---------------------------------------------------
    #   Aperture extraction
    # ---------------------------------------------------
    ap_spec1d, ap_var1d = np.zeros((corrected_data.shape[0], len(xpos))), np.zeros((corrected_data.shape[0], len(xpos)))
    for inte in tqdm(range( ap_spec1d.shape[0] )):
        ap_spec1d[inte, :], ap_var1d[inte, :] = aperture_extract(frame=corrected_data[inte,:,xpos[0]:xpos[-1]+1],\
                                                                variance=corrected_errs[inte,:,xpos[0]:xpos[-1]+1]**2,\
                                                                ord_pos=ypos2d[inte,:], ap_rad=aprad_extraction)
        
    # ------ Calculating the median spectrum across all integrations ------
    med_ap_spec1d = np.nanmedian(ap_spec1d, axis=0)

    # ------- Calculating the residuals of the spectra with respect to the median spectrum ------
    res_ap_spec1d = ap_spec1d - med_ap_spec1d

    # ------ Calculating the threshold for identifying outliers in the residual spectra ------
    limit = np.nanmedian(res_ap_spec1d, axis=0) + 5 * pipe_mad( res_ap_spec1d, axis=0 )

    # ------ Creating a mask for outliers in the residual spectra ------
    mask_outliers = np.ones(res_ap_spec1d.shape, dtype=bool)
    for i in range(res_ap_spec1d.shape[0]):
        mask_outliers[i, :] = np.where(np.abs(res_ap_spec1d[i, :]) > limit, False, True)

    # ------ Replacing the outliers in the original spectra with median spectrum ------
    ap_spec1d_clipped = np.copy(ap_spec1d)
    for i in range(ap_spec1d_clipped.shape[0]):
        ap_spec1d_clipped[i, :] = np.where(mask_outliers[i, :], ap_spec1d_clipped[i, :], med_ap_spec1d)

    # ---------------------------------------------------
    #  Fitting a univariate spline to the spectral trace
    # ---------------------------------------------------

    data1d = SingleOrderPSF(frame=corrected_data[:,:,xpos[0]:xpos[-1]+1],\
                            variance=corrected_errs[:,:,xpos[0]:xpos[-1]+1]**2,\
                            ord_pos=ypos2d, ap_rad=aprad_psf, mask=mask_bcr[:,:,xpos[0]:xpos[-1]+1],\
                            spec=ap_spec1d_clipped)
    psf_frame1d, psf_spline1d, msk_updated_1d = data1d.univariate_psf_frame(niters=3, oversample=oversample_psf, clip=clip_sigma)

    ## Plotting the 1D PSF and the spline fit
    ts1 = np.linspace(np.min(data1d.norm_array[:,0]), np.max(data1d.norm_array[:,0]), 1000)
    msk1 = np.asarray(data1d.norm_array[:,4], dtype=bool)
    msk2 = ( msk1 * msk_updated_1d  ) + ( ~msk1 * ~msk_updated_1d  )
    msk3 = msk1 * msk_updated_1d

    # ---------------------------------------------------
    #  Plotting the fitted PSF
    # ---------------------------------------------------

    plt.figure(figsize=(16/1.5, 9/1.5))
    plt.errorbar(data1d.norm_array[:,0], data1d.norm_array[:,1], fmt='.', color='dodgerblue', markersize=1., zorder=5, label='All points')
    plt.errorbar(data1d.norm_array[~msk1,0], data1d.norm_array[~msk1,1], fmt='.', color='orangered', alpha=0.3, markersize=1., zorder=5, label='Default badpixels')
    plt.errorbar(data1d.norm_array[~msk2,0], data1d.norm_array[~msk2,1], fmt='.', color='darkviolet', alpha=0.3, markersize=1., zorder=5, label='Pixels masked by spline fit')
    plt.plot(ts1, psf_spline1d(ts1), c='k', lw=1., zorder=10, label='Estimated PSF')
    plt.xlabel('Distance from the trace')
    plt.ylabel('Normalised flux')
    plt.legend()
    plt.savefig(pout2 + '/Figs/PSF_fit_' + seg + '.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(16/1.5, 9/1.5))
    plt.errorbar(data1d.norm_array[msk3,0], data1d.norm_array[msk3,1], fmt='.', color='dodgerblue', markersize=1., zorder=5, label='Remaining points')
    plt.plot(ts1, psf_spline1d(ts1), c='k', lw=1., zorder=10, label='Estimated PSF')
    plt.text(-7, 0.5, 'Cleaned version,\nafter removing outliers')
    plt.xlabel('Distance from the trace')
    plt.ylabel('Normalised flux')
    plt.legend()
    plt.savefig(pout2 + '/Figs/PSF_fit_cleaned_' + seg + '.png', bbox_inches='tight')
    plt.close()

    # ----------------------------------------------------------
    #     Updating the bad-pixel map
    # ----------------------------------------------------------

    msk_2d = data1d.table2frame(msk_updated_1d)
    mask_badpix_updated = np.copy(mask_bcr)
    mask_badpix_updated[:,:,xpos[0]:xpos[-1]+1] = mask_bcr[:,:,xpos[0]:xpos[-1]+1] * msk_2d

    # ----------------------------------------------------------
    #     And the optimal extraction
    # ----------------------------------------------------------

    spec1d, var1d = np.zeros((psf_frame1d.shape[0], psf_frame1d.shape[2])), np.zeros((psf_frame1d.shape[0], psf_frame1d.shape[2]))
    syth1d = np.zeros(psf_frame1d.shape)
    for inte in tqdm(range(spec1d.shape[0])):
        spec1d[inte,:], var1d[inte,:], syth1d[inte,:,:] = optimal_extract(psf_frame=psf_frame1d[inte,:,:],\
                                                                          data=corrected_data[inte,:,xpos[0]:xpos[-1]+1],\
                                                                          variance=corrected_errs[inte,:,xpos[0]:xpos[-1]+1]**2,\
                                                                          mask=mask_badpix_updated[inte,:,xpos[0]:xpos[-1]+1],\
                                                                          ord_pos=ypos2d[inte,:], ap_rad=aprad_extraction)
        
    # ----------------------------------------------------------
    #    Plotting the extracted spectrum for all integrations
    # ----------------------------------------------------------

    plt.figure(figsize=(15,5))
    for i in range(spec1d.shape[0]):
        plt.plot(xpos, spec1d[i,:], 'k', alpha=0.1, lw=0.7)
    plt.xlabel('Column number')
    plt.ylabel('#')
    plt.title('Timeseries of spectra')
    plt.savefig(pout2 + '/Figs/Extracted_spectra_' + seg + '_1D.png', bbox_inches='tight')
    plt.close()

    # ------ Calculating the median spectrum across all integrations ------
    med_spec1d = np.nanmedian(spec1d, axis=0)

    # ------ Creating a mask for NaN in spectra ------
    mask_outliers = np.ones(spec1d.shape, dtype=bool)
    for i in range(spec1d.shape[0]):
        mask_outliers[i, :] = np.where(np.isnan(spec1d[i, :]), False, True)

    # ------ Replacing the outliers in the original spectra with median spectrum ------
    spec1d_clipped = np.copy(spec1d)
    for i in range(spec1d_clipped.shape[0]):
        spec1d_clipped[i, :] = np.where(mask_outliers[i, :], spec1d_clipped[i, :], med_spec1d)

    # ---------------------------------------------------
    #  Fitting a bivariate spline to the spectral trace
    # ---------------------------------------------------
    data2 = SingleOrderPSF(frame=corrected_data[:,:,xpos[0]:xpos[-1]+1],\
                           variance=corrected_errs[:,:,xpos[0]:xpos[-1]+1]**2,\
                           ord_pos=ypos2d, ap_rad=aprad_psf, mask=mask_badpix_updated[:,:,xpos[0]:xpos[-1]+1],\
                           spec=spec1d_clipped)
    psf_frame2d, psf_spline2d, msk_after2d = data2.bivariate_psf_frame(niters=3, oversample=oversample_psf, knot_col=10, clip=clip_sigma)

    # --------- Plotting the 2D PSF and the spline fit ---------
    ncol = np.random.choice(xpos)
    des_pts, cont_pts = utils.spln2d_func(ncol1=ncol-xpos[0], datacube=data2)
    fits_2d = psf_spline2d(cont_pts[0], cont_pts[1], grid=False)

    plt.figure(figsize=(16/1.5,9/1.5))
    plt.errorbar(des_pts[0], des_pts[2], fmt='.')
    plt.plot(cont_pts[0], fits_2d, 'k-')
    plt.plot(des_pts[0], psf_spline2d(des_pts[0], des_pts[1], grid=False), 'k.')
    plt.axvline(0., color='k', ls='--')
    plt.title('All frames, for Column ' + str(ncol))
    plt.xlabel('Distance from the trace')
    plt.ylabel('Normalised flux')
    plt.savefig(pout2 + '/Figs/PSF_fit_2D_' + seg + '_col' + str(ncol) + '.png', bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------
    #  Generating the white-light light curve with the best aperture size
    # -------------------------------------------------------
    msk_2d2d = data2.table2frame(msk_after2d)
    mask_badpix_updated2d = np.copy(mask_badpix_updated)
    mask_badpix_updated2d[:,:,xpos[0]:xpos[-1]+1] = mask_badpix_updated[:,:,xpos[0]:xpos[-1]+1] * msk_2d2d

    spec1d, var1d = np.zeros((psf_frame2d.shape[0], psf_frame2d.shape[2])), np.zeros((psf_frame2d.shape[0], psf_frame2d.shape[2]))
    syth1d = np.zeros(psf_frame2d.shape)
    for inte in tqdm(range(spec1d.shape[0])):
        spec1d[inte,:], var1d[inte,:], syth1d[inte,:,:] = optimal_extract(psf_frame=psf_frame2d[inte,:,:],\
                                                                          data=corrected_data[inte,:,xpos[0]:xpos[-1]+1],\
                                                                          variance=corrected_errs[inte,:,xpos[0]:xpos[-1]+1]**2,\
                                                                          mask=mask_badpix_updated2d[inte,:,xpos[0]:xpos[-1]+1],\
                                                                          ord_pos=ypos2d[inte,:], ap_rad=aprad_extraction)
        
    resid1 = np.zeros(syth1d.shape)
    for j in range(resid1.shape[0]):
        resid1[j,:,:] = corrected_data[j,:,xpos[0]:xpos[-1]+1] - syth1d[j,:,:]

    # -------------------------------------------------------
    #   Saving the extracted spectra and the residuals for later use
    # -------------------------------------------------------
    np.savez(pout2 + '/Extracted_spectra_' + seg + '.npz', spec1d=spec1d, var1d=var1d, resid1=resid1, times_bjd=times_bjd)

    t2 = time.time()
    print('Time taken for segment ' + seg + ': {:.2f} minutes'.format((t2-t1)/60))