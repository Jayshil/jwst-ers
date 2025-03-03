import numpy as np
import matplotlib.pyplot as plt
import os
import utils
from stark import SingleOrderPSF, optimal_extract
from tqdm import tqdm
from pathlib import Path
from poetss import poetss
import pickle
import time

# This file is to extract spectrum from all segments
visit = 'NRSPR'

# Input and Output paths
p1 = os.getcwd()
pin = p1 + '/RateInts/Corr_' + visit
pout = p1 + '/NRSPR/Outputs'
if not Path(pout + '/Figures').exists():
    os.mkdir(pout + '/Figures')

## Segment!!!
segs = []
for i in range(4):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))
xstart, xend = 25, 470
aprad = 7.


for se in range(len(segs)):
    print('>>>> --- --- --- --- --- --- --- --- --- ---')
    print('>>>> --- Working on Segment ' + str(segs[se]))
    print('>>>> --- --- --- --- --- --- --- --- --- ---')
    t1 = time.time()
    # See if outputs are there or not
    f1 = Path(pout + '/Spectrum_cube_' + visit + '_' + str(se+1) + '.pkl')
    f2 = Path(pout + '/PSF_data_' + visit + '_' + str(se+1) + '.pkl')
    f3 = Path(pout + '/Dataset_' + visit + '_' + str(se+1) + '.stk')

    if f1.exists() and f2.exists() and f3.exists():
        all_completed = True
    else:
        all_completed = False

    if not all_completed:

        # --------------------------------------------------------------------------
        #
        #                       Loading the data
        #
        # --------------------------------------------------------------------------

        print('>>>> --- Loading the data...')
        corrected_data = np.load(pin + '/Corrected_data_seg' + segs[se] + '.npy')
        corrected_errs = np.load(pin + '/Corrected_errors_seg' + segs[se] + '.npy')
        mask_bcr = np.load(pin + '/Mask_bcr_seg' + segs[se] + '.npy')
        time_bjd = np.load(pin + '/Times_bjd_seg' + segs[se] + '.npy')
        time_bjd = time_bjd + 2400000.5
        nint = np.random.randint(0, corrected_data.shape[0])
        print('>>>> --- Done!!')

        if se == 0:
            # Integration 1111 is bad! So, let's just remove it!
            bad_int = np.ones(corrected_data.shape[0], dtype=bool)
            bad_int[1111] = False
            corrected_data, corrected_errs = corrected_data[bad_int,:,:], corrected_errs[bad_int,:,:]
            mask_bcr, time_bjd = mask_bcr[bad_int,:,:], time_bjd[bad_int]

        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        #
        #                       Centroid positions
        #
        # --------------------------------------------------------------------------

        print('>>>> --- Centroid position computation...')

        def pipe_mad(data):
            return np.nanmedian(np.abs(np.diff(data, axis=0)), axis=0)

        # Finding trace
        cent_cub1 = poetss.find_trace_cof(clean_cube=corrected_data[:,:,xstart:xend], margin=5)
        trace1, dx1 = poetss.fit_multi_trace(cent_mat=cent_cub1, deg=3, clip=3)
        xpos = np.arange(xstart, xend, 1)

        ypos2d = np.zeros((corrected_data.shape[0], len(xpos)))
        for i in range(ypos2d.shape[0]):
            ypos2d[i,:] = trace1 + dx1[i]

        plt.figure(figsize=(15,5))
        im = plt.imshow(corrected_data[nint,:,:], interpolation='none', aspect='auto', cmap='plasma')
        im.set_clim([0,5e3])
        plt.plot(xpos, trace1, 'k-')
        plt.title('Example data with the location of spectral trace')
        plt.savefig(pout + '/Figures/centroid_position_' + str(se+1) + '.png', dpi=500)

        plt.figure(figsize=(15,5))
        plt.plot(np.arange(len(dx1)), dx1, 'k-')
        plt.ylim([-0.015, 0.015])
        plt.xlim([0, corrected_data.shape[0]])
        plt.xlabel('Integration number')
        plt.ylabel('Jitter')
        plt.savefig(pout + '/Figures/centroid_jitter_' + str(se+1) + '.png', dpi=500)

        print('>>>> --- Done!!')

        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        #
        #                       1D PSF
        #
        # --------------------------------------------------------------------------

        print('>>>> --- Fitting 1D PSF to the data...')

        data1d = SingleOrderPSF(frame=corrected_data[:,:,xpos[0]:xpos[-1]+1],\
                                variance=corrected_errs[:,:,xpos[0]:xpos[-1]+1]**2,\
                                ord_pos=ypos2d, ap_rad=aprad, mask=mask_bcr[:,:,xpos[0]:xpos[-1]+1])
        psf_frame1d, psf_spline1d, msk_updated_1d = data1d.univariate_psf_frame(niters=3, oversample=2, clip=10)

        print('>>>> --- Done!!')

        ts1 = np.linspace(np.min(data1d.norm_array[:,0]), np.max(data1d.norm_array[:,0]), 1000)
        msk1 = np.asarray(data1d.norm_array[:,4], dtype=bool) * msk_updated_1d
        plt.figure(figsize=(16/1.5, 9/1.5))
        plt.errorbar(data1d.norm_array[msk1,0], data1d.norm_array[msk1,1], fmt='.')
        plt.plot(ts1, psf_spline1d(ts1), c='k', lw=2., zorder=10)
        plt.xlabel('Distance from the trace')
        plt.ylabel('Normalised flux')
        plt.savefig(pout + '/Figures/1d_psf_fitting_' + str(se+1) + '.png', dpi=500)

        msk_2d = data1d.table2frame(msk_updated_1d)
        mask_badpix_updated = np.copy(mask_bcr)
        mask_badpix_updated[:,:,xpos[0]:xpos[-1]+1] = mask_bcr[:,:,xpos[0]:xpos[-1]+1] * msk_2d

        print('>>>> --- Spectrum from 1D PSF...')
        spec1d, var1d = np.zeros((psf_frame1d.shape[0], psf_frame1d.shape[2])), np.zeros((psf_frame1d.shape[0], psf_frame1d.shape[2]))
        syth1d = np.zeros(psf_frame1d.shape)
        for inte in tqdm(range(spec1d.shape[0])):
            spec1d[inte,:], var1d[inte,:], syth1d[inte,:,:] = optimal_extract(psf_frame=psf_frame1d[inte,:,:],\
                                                                            data=corrected_data[inte,:,xpos[0]:xpos[-1]+1],\
                                                                            variance=corrected_errs[inte,:,xpos[0]:xpos[-1]+1]**2,\
                                                                            mask=mask_badpix_updated[inte,:,xpos[0]:xpos[-1]+1],\
                                                                            ord_pos=ypos2d[inte,:], ap_rad=aprad)
        print('>>>> --- Done!!')

        print('>>>> --- Computing white-light lightcurve...')
        wht_light_lc, wht_light_err = poetss.white_light(spec1d, np.sqrt(var1d))
        print('++++ --- MAD of the white-light light curve is: {:.2f} ppm'.format(pipe_mad(wht_light_lc/np.nanmedian(wht_light_lc))*1e6))
        print('>>>> --- Done!!')

        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        #
        #                       2D PSF
        #
        # --------------------------------------------------------------------------

        print('>>>> --- Fitting 2D PSF to the data...')
        data2 = SingleOrderPSF(frame=corrected_data[:,:,xpos[0]:xpos[-1]+1],\
                            variance=corrected_errs[:,:,xpos[0]:xpos[-1]+1]**2,\
                            ord_pos=ypos2d, ap_rad=aprad, mask=mask_badpix_updated[:,:,xpos[0]:xpos[-1]+1],\
                            spec=spec1d)
        psf_frame2d, psf_spline2d, msk_after2d = data2.bivariate_psf_frame(niters=3, oversample=2, knot_col=10, clip=10)
        print('>>>> --- Done!!')

        ncol = np.random.choice(xpos)
        des_pts, cont_pts = utils.spln2d_func(ncol1=ncol-xpos[0], datacube=data2)
        fits_2d = psf_spline2d(cont_pts[0], cont_pts[1], grid=False)

        # --------- Saving a plot showing how good the spline fitting is
        plt.figure(figsize=(16/1.5,9/1.5))
        plt.errorbar(des_pts[0], des_pts[2], fmt='.')
        plt.plot(cont_pts[0], fits_2d, 'k-')
        plt.plot(des_pts[0], psf_spline2d(des_pts[0], des_pts[1], grid=False), 'k.')
        plt.axvline(0., color='k', ls='--')
        plt.title('All frames, for Column ' + str(ncol))
        plt.xlabel('Distance from the trace')
        plt.ylabel('Normalised flux')
        plt.savefig(pout + '/Figures/2d_spline_' + str(se+1) + '.png', dpi=500)

        # ---------- PSF change with wavelength
        # Defining pixel coordinates
        pix_cor_res = 50000
        pix_corr = np.linspace(-7., 7., pix_cor_res)

        cols = xpos - xpos[0]
        max_amp = np.zeros(len(cols))
        fwhm = np.zeros(len(cols))

        for i in range(len(cols)):
            fit2 = psf_spline2d(x=pix_corr, y=np.ones(pix_cor_res)*cols[i], grid=False)
            # Maximum amplitude
            max_amp[i] = np.max(fit2)
            # Maximum amplitude location
            idx_max_amp = np.where(fit2 == np.max(fit2))[0][0]
            # fwhm
            hm = (np.max(fit2) + np.min(fit2))/2
            idx_hm = np.where(np.abs(fit2 - hm)<0.005)[0]
            idx_hm_up, idx_hm_lo = 0, 0
            diff_up1, diff_lo1 = 10., 10.
            for j in range(len(idx_hm)):
                if idx_hm[j] > idx_max_amp:
                    diff_u1 = np.abs(fit2[idx_hm[j]] - hm)
                    if diff_u1 < diff_up1:
                        diff_up1 = diff_u1
                        idx_hm_up = idx_hm[j]
                else:
                    diff_l1 = np.abs(fit2[idx_hm[j]] - hm)
                    if diff_l1 < diff_lo1:
                        diff_lo1 = diff_l1
                        idx_hm_lo = idx_hm[j]
            fwhm[i] = np.abs(pix_corr[idx_hm_up] - pix_corr[idx_hm_lo])

        fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=True, facecolor='white')

        axs[0].plot(xpos, max_amp, 'k-')
        axs[0].set_ylabel('Maximum Amplitude', fontsize=14)

        axs[1].plot(xpos, fwhm, 'k-')
        axs[1].set_ylabel('FWHM', fontsize=14)
        axs[1].set_xlabel('Column number', fontsize=14)

        axs[1].set_xlim([xpos[0], xpos[-1]])

        axs[0].set_title('PSF evolution with wavelength', fontsize=15)

        plt.setp(axs[0].get_yticklabels(), fontsize=12)
        plt.setp(axs[1].get_xticklabels(), fontsize=12)
        plt.setp(axs[1].get_yticklabels(), fontsize=12)

        plt.tight_layout()
        plt.savefig(pout + '/Figures/psf_variation_' + str(se+1) + '.png', dpi=500)

        # ------- Updating the mask
        msk_2d2d = data2.table2frame(msk_after2d)
        mask_badpix_updated2d = np.copy(mask_badpix_updated)
        mask_badpix_updated2d[:,:,xpos[0]:xpos[-1]+1] = mask_badpix_updated[:,:,xpos[0]:xpos[-1]+1] * msk_2d2d

        print('>>>> --- Extracting the spectrum using the fitted PSF...')
        min_scat_ap = 5.

        spec1d, var1d = np.zeros((psf_frame2d.shape[0], psf_frame2d.shape[2])), np.zeros((psf_frame2d.shape[0], psf_frame2d.shape[2]))
        syth1d = np.zeros(psf_frame2d.shape)
        for inte in tqdm(range(spec1d.shape[0])):
            spec1d[inte,:], var1d[inte,:], syth1d[inte,:,:] = optimal_extract(psf_frame=psf_frame2d[inte,:,:],\
                                                                            data=corrected_data[inte,:,xpos[0]:xpos[-1]+1],\
                                                                            variance=corrected_errs[inte,:,xpos[0]:xpos[-1]+1]**2,\
                                                                            mask=mask_badpix_updated2d[inte,:,xpos[0]:xpos[-1]+1],\
                                                                            ord_pos=ypos2d[inte,:], ap_rad=min_scat_ap)
        print('>>>> --- Done!!')

        plt.figure(figsize=(15,5))
        for i in range(spec1d.shape[0]):
            plt.plot(xpos, spec1d[i,:], 'k', alpha=0.1)
        plt.xlabel('Column number')
        plt.ylabel('#')
        plt.title('Timeseries of spectra')
        plt.savefig(pout + '/Figures/timeseries_spectra_' + str(se+1) + '.png', dpi=500)

        print('>>>> --- Computing residual frame...')
        resid1 = np.zeros(syth1d.shape)
        for j in range(resid1.shape[0]):
            resid1[j,:,:] = corrected_data[j,:,xpos[0]:xpos[-1]+1] - syth1d[j,:,:]

        med_resid = np.nanmedian(resid1, axis=0)

        plt.figure(figsize=(15,5))
        im = plt.imshow(med_resid, interpolation='none')#, aspect='auto')
        im.set_clim([-5,5])
        plt.title('Median residual frame')
        plt.savefig(pout + '/Figures/med_resid_' + str(se+1) + '.png', dpi=500)

        print('>>>> --- Done!!')


        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        #
        #                Computing the white-light light curve
        #
        # --------------------------------------------------------------------------


        print('>>>> --- Computing white-light lightcurve...')
        wht_light_lc, wht_light_err = poetss.white_light(spec1d, np.sqrt(var1d))
        print('++++ --- MAD of the white-light light curve is: {:.2f} ppm'.format(pipe_mad(wht_light_lc/np.nanmedian(wht_light_lc))*1e6))

        plt.figure(figsize=(15,5))
        plt.errorbar(time_bjd, wht_light_lc/np.nanmedian(wht_light_lc), \
                    yerr=wht_light_err/np.nanmedian(wht_light_lc), fmt='.', c='k')
        plt.title('White-light lightcurve, MAD: {:.4f} ppm'.format(pipe_mad(wht_light_lc/np.nanmedian(wht_light_lc)) * 1e6))
        plt.xlabel('Time (BJD)')
        plt.ylabel('Relative flux')
        plt.ylim([0.9,1.1])
        plt.savefig(pout + '/Figures/white_light_lc_' + str(se+1) + '.png', dpi=500)
        print('>>>> --- Done!!')

        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        #
        #                       Saving the data
        #
        # --------------------------------------------------------------------------

        print('>>>> --- Saving the results...')
        # Dictionary for saving the final results
        dataset = {}
        dataset['spectra'] = spec1d
        dataset['variance'] = var1d
        dataset['resid'] = resid1
        dataset['times'] = time_bjd
        pickle.dump(dataset, open(pout + '/Spectrum_cube_' + visit + '_' + str(se+1) + '.pkl','wb'))

        psf_res = {}
        psf_res['psf'] = psf_frame2d
        psf_res['spline'] = psf_spline2d
        pickle.dump(psf_res, open(pout + '/PSF_data_' + visit + '_' + str(se+1) + '.pkl','wb'))

        pickle.dump(data2, open(pout + '/Dataset_' + visit + '_' + str(se+1) + '.stk','wb'))
        print('>>>> --- Done!!')

        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
    else:
        print('>>>> --- All is already done!!')
        print('>>>> --- Moving on to the next segment!')

    t2 = time.time()
    print('>>>> --- Total time taken: {:.4f} minutes'.format((t2-t1)/60))
    print('>>>> --- --- --- --- --- --- --- --- --- ---')
    print('>>>> ---  Segment ' + str(segs[se]) + ' analysis completed!!')
    print('>>>> --- --- --- --- --- --- --- --- --- ---')