import numpy as np
import matplotlib.pyplot as plt
import os
from stark import SingleOrderPSF, optimal_extract
import pickle
from tqdm import tqdm
import time
from path import Path
from poetss import poetss
import matplotlib
import matplotlib.cm as cm

# This file is to extract spectra from all segments
# NIRCam and NIRSpec are original extraction, _v2 folder contains extraction when median residual image was
# subtracted from the data

## Segment!!!
segs = []
for i in range(4):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))
segs = ['004']
## Setting the aperture size!
aprad = 7.
nint, ncol = 100, 150

visit = 'NIRSpec'
version = None         # Set this to None if this is the first analysis

if visit == 'NIRCam':
    xstart, xend = 30, 1650
elif visit == 'NIRSpec':
    xstart, xend = 40, 450
else:
    print('Something is wrong...')

# Loading and correcting the data
if version == None:
    p2 = os.getcwd() + '/stark/Outputs/' + visit
else:
    p2 = os.getcwd() + '/stark/Outputs/' + visit + '_' + version
    p2_1 = os.getcwd() + '/stark/Outputs/' + visit
if not Path(p2 + '/Figures').exists():
    os.mkdir(p2 + '/Figures')
p4 = os.getcwd() + '/RateInts/Corr_' + visit

# Computing median residual image
if version is not None:
    print('>>>> --- Computing median residual image...')
    med_resid_all_seg = []
    for i in tqdm(range(len(segs))):
        post1 = pickle.load(open(p2_1 + '/Spectrum_cube_seg' + segs[i] + '_' + visit + '.pkl', 'rb'))
        resid1 = post1['resid']
        med_resid_all_seg.append(np.median(resid1, axis=0))
    med_resid_all_seg1 = np.dstack(med_resid_all_seg)
    med_resid_img = np.median(med_resid_all_seg1, axis=2)
    print('>>>> --- Done!!')


for seg in range(len(segs)):
    t1 = time.time()
    print('---------------------------------')
    print('Working on Segment: ', segs[seg])
    print('')
    f12 = Path(p2 + '/Spectrum_cube_seg' + segs[seg] + '_' + visit + '.pkl')
    f13 = Path(p2 + '/Traces_seg' + segs[seg] + '_' + visit + '.pkl')
    f14 = Path(p2 + '/PSF_data_seg' + segs[seg] + '_' + visit + '.pkl')
    if f12.exists() and f13.exists() and f14.exists():
        print('>>>> --- The file already exists...')
        print('         Continuing to the next segment...')
        continue
    else:
        pass
    print('>>>> --- Loading the dataset...')
    corrected_data = np.load(p4 + '/Corrected_data_seg' + segs[seg] + '.npy')
    corrected_errs = np.load(p4 + '/Corrected_errors_seg' + segs[seg] + '.npy')
    mask_bcr = np.load(p4 + '/Mask_bcr_seg' + segs[seg] + '.npy')
    times_bjd = np.load(p4 + '/Times_bjd_seg' + segs[seg] + '.npy')
    print('>>>> --- Done!!')

    if version is not None:
        print('>>>> --- Second version of the data...')
        print('>>>> --- Subtracting median residual image from the corrected data...')
        corrected_data[:,:,xstart:xend] = corrected_data[:,:,xstart:xend] - med_resid_img[None,:,:]
        print('>>>> --- Done!!')

    print('>>>> --- Finding trace positions...')
    # CM method
    if version is None:
        cent_cub1 = poetss.find_trace_cof(clean_cube=corrected_data[:,:,xstart:xend], margin=5)
        trace1, dx1 = poetss.fit_multi_trace(cent_mat=cent_cub1, deg=2, clip=3)
        xpos = np.arange(xstart, xend, 1)
        
        ## Saving the results
        traces_pos = {}
        traces_pos['xpos'] = xpos
        traces_pos['median_trace'] = trace1
        traces_pos['jitter'] = dx1
        pickle.dump(traces_pos, open(p2 + '/Traces_seg' + segs[seg] + '_' + visit + '.pkl','wb'))
    else:
        print('>>>> --- Second version of the data, re-using the trace from the first version...')
        print('>>>> --- Loading the trace positions...')
        traces_pos = pickle.load(open(p2_1 + '/Traces_seg' + segs[seg] + '_' + visit + '.pkl', 'rb'))
        xpos, trace1, dx1 = traces_pos['xpos'], traces_pos['median_trace'], traces_pos['jitter']
    
    ypos2d = np.zeros((corrected_data.shape[0], len(xpos)))
    for i in range(ypos2d.shape[0]):
        ypos2d[i,:] = trace1 + dx1[i]
        
    print('>>>> --- Done!!')

    # 1D spline fitting
    print('>>>> --- Initial 1D spline fitting...')
    data1 = SingleOrderPSF(frame=corrected_data[:,:,xpos[0]:xpos[-1]+1],\
                            variance=corrected_errs[:,:,xpos[0]:xpos[-1]+1]**2,\
                            ord_pos=ypos2d, ap_rad=aprad, mask=mask_bcr[:,:,xpos[0]:xpos[-1]+1])
    psf_frame1d, psf_spline1d = data1.univariate_psf_frame(niters=3, oversample=2, clip=10000)
    print('>>>> --- Done!!')
    print('>>>> --- Making a figure...')
    ## Save a figure to make sure that it was working!
    ts1 = np.linspace(np.min(data1.norm_array[:,0]), np.max(data1.norm_array[:,0]), 1000)
    msk1 = np.asarray(data1.norm_array[:,4], dtype=bool)

    plt.figure(figsize=(16/1.5, 9/1.5))
    plt.errorbar(data1.norm_array[msk1,0], data1.norm_array[msk1,1], fmt='.')
    plt.plot(ts1, psf_spline1d(ts1), c='k', lw=2., zorder=10)

    plt.savefig(p2 + '/Figures/1dSpline_Seg' + segs[seg] + '.png', dpi=500)
    print('>>>> --- Done!!')

    print('>>>> --- Finding spectrum using 1D spline...')
    spec1d, var1d = np.zeros((psf_frame1d.shape[0], psf_frame1d.shape[2])), np.zeros((psf_frame1d.shape[0], psf_frame1d.shape[2]))
    syth1d = np.zeros(psf_frame1d.shape)
    for inte in tqdm(range(spec1d.shape[0])):
        spec1d[inte,:], var1d[inte,:], syth1d[inte,:,:] = optimal_extract(psf_frame=psf_frame1d[inte,:,:],\
                                                                            data=corrected_data[inte,:,xpos[0]:xpos[-1]+1],\
                                                                            variance=corrected_errs[inte,:,xpos[0]:xpos[-1]+1]**2,\
                                                                            mask=mask_bcr[inte,:,xpos[0]:xpos[-1]+1],\
                                                                            ord_pos=ypos2d[inte,:], ap_rad=aprad)
    print('>>>> --- Done!!')

    print('>>>> --- 2D spline fitting and spectral extraction...')
    # For 2d spline fitting
    for _ in range(1):
        data2 = SingleOrderPSF(frame=corrected_data[:,:,xpos[0]:xpos[-1]+1],\
                                variance=corrected_errs[:,:,xpos[0]:xpos[-1]+1]**2,\
                                ord_pos=ypos2d, ap_rad=aprad, mask=mask_bcr[:,:,xpos[0]:xpos[-1]+1],\
                                spec=None)
        psf_frame2d, psf_spline2d = data2.bivariate_psf_frame(niters=3, oversample=2, knot_col=10, clip=10000)
        spec1d, var1d = np.zeros((psf_frame2d.shape[0], psf_frame2d.shape[2])), np.zeros((psf_frame2d.shape[0], psf_frame2d.shape[2]))
        syth1d = np.zeros(psf_frame2d.shape)
        for inte in tqdm(range(spec1d.shape[0])):
            spec1d[inte,:], var1d[inte,:], syth1d[inte,:,:] = optimal_extract(psf_frame=psf_frame2d[inte,:,:],\
                                                                                data=corrected_data[inte,:,xpos[0]:xpos[-1]+1],\
                                                                                variance=corrected_errs[inte,:,xpos[0]:xpos[-1]+1]**2,\
                                                                                mask=mask_bcr[inte,:,xpos[0]:xpos[-1]+1],\
                                                                                ord_pos=ypos2d[inte,:], ap_rad=aprad)
    print('>>>> --- Done!!')

    print('>>>> --- Making some figures...')
    ## Saving figures to make sure that 2D spline fitting was working!!
    msk5 = data2.col_array_pos[:,ncol-xpos[0],0]
    npix1 = data2.col_array_pos[:,ncol-xpos[0],1]

    xpoints = np.array([])
    ypoints = np.array([])
    zpoints = np.array([])
    for i in range(len(msk5)):
        xdts = data2.norm_array[msk5[i]:msk5[i]+npix1[i],0]
        ydts = data2.norm_array[msk5[i]:msk5[i]+npix1[i],3]
        zdts = data2.norm_array[msk5[i]:msk5[i]+npix1[i],1]
        xpoints = np.hstack((xpoints, xdts))
        ypoints = np.hstack((ypoints, ydts))
        zpoints = np.hstack((zpoints, zdts))

    xpts1 = np.linspace(np.min(xpoints), np.max(xpoints), 1000)
    ypts1 = np.ones(1000)*ypoints[0]

    fits_2d = psf_spline2d(xpts1, ypts1, grid=False)

    plt.figure(figsize=(16/1.5,9/1.5))
    plt.errorbar(xpoints, zpoints, fmt='.')
    plt.plot(xpts1, fits_2d, 'k-')
    plt.plot(xpoints, psf_spline2d(xpoints, ypoints, grid=False), 'k.')
    plt.axvline(0., color='k', ls='--')
    plt.title('All frames, for a given column')
    plt.savefig(p2 + '/Figures/2dSpline_Seg' + segs[seg] + '.png')

    ## Saving all extracted spectrum in a single file!
    plt.figure(figsize=(15,5))
    for i in range(spec1d.shape[0]):
        plt.plot(xpos, spec1d[i,:], 'k', alpha=0.5)
    plt.savefig(p2 + '/Figures/All_spectra_Seg' + segs[seg] + '.png')

    print('>>>> --- Done!!')

    print('>>>> --- Saving results...')

    # For creating residual image
    resid1 = np.zeros(syth1d.shape)
    for j in range(resid1.shape[0]):
        resid1[j,:,:] = corrected_data[j,:,xpos[0]:xpos[-1]+1] - syth1d[j,:,:]

    # Dictionary for saving the final results
    dataset = {}
    dataset['spectra'] = spec1d
    dataset['variance'] = var1d
    dataset['resid'] = resid1
    dataset['times'] = times_bjd
    pickle.dump(dataset, open(p2 + '/Spectrum_cube_seg' + segs[seg] + '_' + visit + '.pkl','wb'))

    psf_res = {}
    psf_res['psf'] = psf_frame2d
    psf_res['spline'] = psf_spline2d
    pickle.dump(psf_res, open(p2 + '/PSF_data_seg' + segs[seg] + '_' + visit + '.pkl','wb'))
    t2 = time.time()
    print('>>>> --- Done!!')
    print('>>>> --- Total time taken: {:.4f} minutes'.format((t2-t1)/60))


"""dataset_all = {}
for seg in range(len(segs)):
    dta = pickle.load(open(p2 + '/Spectrum_cub_seg_' + segs[seg] + '_V1.pkl', 'rb'))
    dataset_all['Seg' + str(segs[seg])] = {}
    dataset_all['Seg' + str(segs[seg])]['spectra'] = dta['spectra']
    dataset_all['Seg' + str(segs[seg])]['variance'] = dta['variance']
    dataset_all['Seg' + str(segs[seg])]['resid'] = dta['resid']
    dataset_all['Seg' + str(segs[seg])]['times'] = dta['times']
pickle.dump(dataset_all, open(p2 + '/Spectrum_cube_all_seg_V1.pkl','wb'))"""

"""for seg in range(len(segs)):
    os.system('rm -rf ' + p2 + '/Spectrum_cub_seg_' + segs[seg] + '_V1.pkl')"""