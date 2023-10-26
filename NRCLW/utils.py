import numpy as np
from photutils.aperture import CircularAnnulus, CircularAperture, ApertureStats
from photutils.aperture import aperture_photometry as aphot
import warnings

def replace_nan(data, max_iter = 50):
    """Replaces NaN-entries by mean of neighbours.
    Iterates until all NaN-entries are replaced or
    max_iter is reached. Works on N-dimensional arrays.
    """
    nan_data = data.copy()
    shape = np.append([2*data.ndim], data.shape)
    interp_cube = np.zeros(shape)
    axis = tuple(range(data.ndim))
    shift0 = np.zeros(data.ndim, int)
    shift0[0] = 1
    shift = []     # Shift list will be [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for n in range(data.ndim):
        shift.append(tuple(np.roll(-shift0, n)))
        shift.append(tuple(np.roll(shift0, n)))
    for _j in range(max_iter):
        for n in range(2*data.ndim):
            interp_cube[n] = np.roll(nan_data, shift[n], axis = axis)   # interp_cube would be (4, data.shape[0], data.shape[1]) sized array
        with warnings.catch_warnings():                                 # with shifted position in each element (so that we can take its mean)
            warnings.simplefilter('ignore', RuntimeWarning)
            mean_data = np.nanmean(interp_cube, axis=0)
        nan_data[np.isnan(nan_data)] = mean_data[np.isnan(nan_data)]
        if np.sum(np.isnan(nan_data)) == 0:
            break
    return nan_data

def identify_crays(frames, mask_bp, clip=5, niters=5):
    """Given a data cube and bad-pixel map, this function identifies cosmic rays by using median frame"""
    # Masking bad pixels as NaN
    mask_cr = np.copy(mask_bp)
    for _ in range(niters):
        # Flagging bad data as Nan
        frame_new = np.copy(frames)
        frame_new[mask_cr == 0.] = np.nan
        # Median frame
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            median_frame = np.nanmedian(frame_new, axis=0)  # 2D frame
            # Creating residuals
            resids = frame_new - median_frame[None,:,:]
            # Median and std of residuals
            med_resid, std_resid = np.nanmedian(resids, axis=0), np.nanstd(resids, axis=0)
        limit = med_resid + (clip*std_resid)
        mask_cr1 = np.abs(resids) < limit[None,:,:]
        mask_cr = mask_cr1*mask_bp
    return mask_cr

def spln2d_func(ncol1, datacube):
    ## Saving figures to make sure that 2D spline fitting was working!!
    msk5 = datacube.col_array_pos[:,ncol1,0]
    npix1 = datacube.col_array_pos[:,ncol1,1]

    xpoints = np.array([])
    ypoints = np.array([])
    zpoints = np.array([])
    for i in range(len(msk5)):
        xdts = datacube.norm_array[msk5[i]:msk5[i]+npix1[i],0]
        ydts = datacube.norm_array[msk5[i]:msk5[i]+npix1[i],3]
        zdts = datacube.norm_array[msk5[i]:msk5[i]+npix1[i],1]
        msk_bad = np.asarray(datacube.norm_array[msk5[i]:msk5[i]+npix1[i],4], dtype=bool)
        xdts, ydts, zdts = xdts[msk_bad], ydts[msk_bad], zdts[msk_bad]
        xpoints = np.hstack((xpoints, xdts))
        ypoints = np.hstack((ypoints, ydts))
        zpoints = np.hstack((zpoints, zdts))

    xpts1 = np.linspace(np.min(xpoints)-0.1, np.max(xpoints)+0.1, 1000)
    ypts1 = np.ones(1000)*ypoints[0]

    return [xpoints,ypoints,zpoints], [xpts1, ypts1]

def find_center(image, rmin=None, rmax=None, cmin=None, cmax=None):
    """Given the image, this function will find center of the PSF using center-of-flux method"""
    # Row is the first index, column is the second index
    # First find the subimage if min & max row, cols are provided
    if (rmin != None)&(rmax != None)&(cmin == None)&(cmax == None):
        subimg = image[rmin:rmax, :]
    elif (rmin == None)&(rmax == None)&(cmin != None)&(cmax != None):
        subimg = image[:, cmin:cmax]
    elif (rmin != None)&(rmax != None)&(cmin != None)&(cmax != None):
        subimg = image[rmin:rmax, cmin:cmax]
    else:
        subimg = np.copy(image)
    
    # Row and column indices
    row_idx, col_idx = np.arange(subimg.shape[0]), np.arange(subimg.shape[1])

    # And now the center of *subimage*
    cen_r_sub = np.sum(row_idx * np.sum(subimg, axis=1)) / np.sum(subimg.flatten())
    cen_c_sub = np.sum(col_idx * np.sum(subimg, axis=0)) / np.sum(subimg.flatten())

    # This was the center of subimage, let's transform that to image coordinates
    if (rmin != None)&(cmin == None):
        cen_r, cen_c = cen_r_sub + rmin, cen_c_sub
    elif (rmin == None)&(cmin != None):
        cen_r, cen_c = cen_r_sub, cen_c_sub + cmin
    elif (rmin != None)&(cmin != None):
        cen_r, cen_c = cen_r_sub + rmin, cen_c_sub + cmin
    else:
        cen_r, cen_c = cen_r_sub, cen_c_sub
    return cen_r, cen_c

def aperture_mask(image, err, cen_r, cen_c, rad):
    """Given the image, error data, centroids, and aperture radius, this function will generate
    what is called an aperture mask for the data and error array. Aperture mask is an array which will
    have non-zero data points within the aperture radius but zeros everywhere outside of the aperture"""
    # Let's first find a distance array which will contain the distance of each pixels from the center
    idx_arr_r, idx_arr_c = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    idx_arr_r, idx_arr_c = np.transpose(idx_arr_r), np.transpose(idx_arr_c)
    # Both of above array would be of the dimension of the image array
    # idx_arr_r[row, col] = row index and idx_arr_c[row, col] = col index

    # Distance array would give distace of each pixel from the center
    distance = np.sqrt(((idx_arr_r - cen_r)**2) + ((idx_arr_c - cen_c)**2))

    # Now, creating mask
    msk = np.zeros(image.shape)
    msk[distance < rad] = 1.

    # And, aperture mask
    aper_mask, err_mask = image * msk, err * msk

    return aper_mask, err_mask, msk


def sky_mask(image, err, cen_r, cen_c, rad1, rad2):
    """Given the image, error data, centroids and inner and outer radii of the sky annulus, this function
    will generate a sky background mask for the data and error array. Sky mask is an array which will
    have non-zero data points only within the sky annulus but zero elsewhere"""
    # Let's first find a distance array which will contain the distance of each pixels from the center
    idx_arr_r, idx_arr_c = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
    idx_arr_r, idx_arr_c = np.transpose(idx_arr_r), np.transpose(idx_arr_c)
    # Both of above array would be of the dimension of the image array
    # idx_arr_r[row, col] = row index and idx_arr_c[row, col] = col index

    # Distance array would give distace of each pixel from the center
    distance = np.sqrt(((idx_arr_r - cen_r)**2) + ((idx_arr_c - cen_c)**2))

    # Now, creating mask
    msk = np.zeros(image.shape)
    msk[(distance > rad1)&(distance < rad2)] = 1.

    # And, aperture mask
    sky_mask, err_mask = image * msk, err * msk

    return sky_mask, err_mask, msk

def aperture_photometry(image, err, cen_r, cen_c, rad, sky_rad1, sky_rad2, method='manual', **kwargs):
    """Given the image and error array, along with centroids and aperture radius as well as inner and
    outer radii of the sky annulus, this function will give you the aperture photometry and errors on it"""
    # First computing centers
    if method == 'manual':
        if (sky_rad1 != None)&(sky_rad2 != None):
            # Let's first obtain sky background flux per pixel (If sky radii are not None)
            sky_msk3, sky_err3, msk3 = sky_mask(image, err, cen_r=cen_r, cen_c=cen_c, rad1=sky_rad1, rad2=sky_rad2)
            
            ## Expression of Sky flux per pixel and sky error per pixel
            sky_flx_per_pix = np.sum(sky_msk3) / np.sum(msk3)
            sky_flx_err_per_pix = np.sqrt(np.sum(sky_err3**2)) / np.sum(msk3)

            # Now, the aperture flux
            flx1, err1, mask8 = aperture_mask(image, err, cen_r=cen_r, cen_c=cen_c, rad=rad)
            tot_sky_bkg = np.sum(mask8)*sky_flx_per_pix
            ape_flx = np.sum(flx1) - tot_sky_bkg
            ape_err = np.sqrt(np.sum(err1**2) + (np.sum(mask8) * sky_flx_err_per_pix * sky_flx_err_per_pix))
        else:
            # Simply computing the aperture flux from aperture mask if sky radii are None
            flx1, err1, mask8 = aperture_mask(image, err, cen_r=cen_r, cen_c=cen_c, rad=rad)
            tot_sky_bkg = 0.
            ape_flx = np.sum(flx1)
            ape_err = np.sqrt(np.sum(err1**2))
    elif method == 'photutils':
        if (sky_rad1 != None)&(sky_rad2 != None):
            # First let's perform the background subtraction
            sky_aper = CircularAnnulus((int(cen_c), int(cen_r)), r_in=sky_rad1, r_out=sky_rad2)
            sky_aperstats = ApertureStats(image, sky_aper)
            bkg_mean, bkg_std = sky_aperstats.mean , sky_aperstats.std   # Mean sky background per pixel
            
            # Now computing the aperture flux
            circ_aper = CircularAperture((cen_c, cen_r), r=rad)
            ap_phot = aphot(data=image, apertures=circ_aper, error=err, **kwargs)
            tot_sky_bkg = bkg_mean * circ_aper.area_overlap(image)
            phot_bkgsub = ap_phot['aperture_sum'] - tot_sky_bkg  # Background subtraction

            ## Error estimation in background subtracted photometry
            phot_bkgsub_err = np.sqrt((ap_phot['aperture_sum_err']**2) + (circ_aper.area_overlap(image) * bkg_std * bkg_std))

            ## Results
            ape_flx, ape_err = phot_bkgsub[0], phot_bkgsub_err[0]
        else:
            # Simply computing the flux inside an aperture 
            circ_aper = CircularAperture((cen_c, cen_r), r=rad)
            ap_phot = aphot(data=image, apertures=circ_aper, error=err, **kwargs)
            ape_flx, ape_err = ap_phot['aperture_sum'][0], ap_phot['aperture_sum_err'][0]
            tot_sky_bkg = 0.
    else:
        raise Exception('Please enter correct method...\nMethod can either be manual or photutils.')
        
    return ape_flx, ape_err, tot_sky_bkg

def lcbin(time, flux, binwidth=0.06859, nmin=4, time0=None,
        robust=False, tmid=False):
    """
    This code is taken from the code `pycheops`
    Calculate average flux and error in time bins of equal width.
    The default bin width is equivalent to one CHEOPS orbit in units of days.
    To avoid binning data on either side of the gaps in the light curve due to
    the CHEOPS orbit, the algorithm searches for the largest gap in the data
    shorter than binwidth and places the bin edges so that they fall at the
    centre of this gap. This behaviour can be avoided by setting a value for
    the parameter time0.
    The time values for the output bins can be either the average time value
    of the input points or, if tmid is True, the centre of the time bin.
    If robust is True, the output bin values are the median of the flux values
    of the bin and the standard error is estimated from their mean absolute
    deviation. Otherwise, the mean and standard deviation are used.
    The output values are as follows.
    * t_bin - average time of binned data points or centre of time bin.
    * f_bin - mean or median of the input flux values.
    * e_bin - standard error of flux points in the bin.
    * n_bin - number of flux points in the bin.
    :param time: time
    :param flux: flux (or other quantity to be time-binned)
    :param binwidth:  bin width in the same units as time
    :param nmin: minimum number of points for output bins
    :param time0: time value at the lower edge of one bin
    :param robust: use median and robust estimate of standard deviation
    :param tmid: return centre of time bins instead of mean time value
    :returns: t_bin, f_bin, e_bin, n_bin
    """
    if time0 is None:
        tgap = (time[1:]+time[:-1])/2
        gap = time[1:]-time[:-1]
        j = gap < binwidth
        gap = gap[j]
        tgap = tgap[j]
        time0 = tgap[np.argmax(gap)]
        time0 = time0 - binwidth*np.ceil((time0-min(time))/binwidth)

    n = np.int(1+np.ceil(np.ptp(time)/binwidth))
    r = (time0,time0+n*binwidth)
    n_in_bin,bin_edges = np.histogram(time,bins=n,range=r)
    bin_indices = np.digitize(time,bin_edges)

    t_bin = np.zeros(n)
    f_bin = np.zeros(n)
    e_bin = np.zeros(n)
    n_bin = np.zeros(n, dtype=np.int)

    for i,n in enumerate(n_in_bin):
        if n >= nmin:
            j = bin_indices == i+1
            n_bin[i] = n
            if tmid:
                t_bin[i] = (bin_edges[i]+bin_edges[i+1])/2
            else:
                t_bin[i] = np.nanmean(time[j])
            if robust:
                f_bin[i] = np.nanmedian(flux[j])
                e_bin[i] = 1.25*np.nanmean(abs(flux[j] - f_bin[i]))/np.sqrt(n)
            else:
                f_bin[i] = np.nanmean(flux[j])
                e_bin[i] = np.std(flux[j])/np.sqrt(n-1)

    j = (n_bin >= nmin)
    return t_bin[j], f_bin[j], e_bin[j], n_bin[j]

def roeba_backgroud_sub(frame, mask, fast_read=True, row_by_row=True):
    """This function does the background subtraction on one integration based on the ROEBA
    algorithm (Schlawin et al. 2023). All non-zero pixels in mask are used in computation"""
    # First let's replace all masked pixels with NaN so we can use nanmedian etc.
    data_frame, bkg_sub_frame = np.copy(frame), np.copy(frame)
    bkg_sub_frame[mask == 0.] = np.nan

    ## First amplifier ROEBA
    ### Slow-read correction (i.e., along the columns)
    first_amp_roeba, first_amp_data = np.zeros(bkg_sub_frame[:,0:512].shape), np.copy(bkg_sub_frame[:,0:512])
    first_amp_roeba[:,::2] = np.nanmedian(first_amp_data[:,::2])   # Computing median of all even columns and storing it
    first_amp_roeba[:,1::2] = np.nanmedian(first_amp_data[:,1::2]) # Computing median of all odd columns and storing it
    data_frame[:,0:512] = data_frame[:,0:512] - first_amp_roeba
    if fast_read:
        ### Fast-read correction (i.e., along the rows)
        dummy_frame = np.copy(data_frame)
        dummy_frame[mask == 0.] = np.nan
        if row_by_row:
            data_frame[:,0:512] = data_frame[:,0:512] - np.nanmedian(dummy_frame[:,0:512], axis=1)[:,None]
        else:
            data_frame[:,0:512] = data_frame[:,0:512] - np.nanmedian(dummy_frame[:,0:512])

    ## Second amplifier ROEBA
    ### Slow-read correction (i.e., along the columns)
    second_amp_roeba, second_amp_data = np.zeros(bkg_sub_frame[:,512:1024].shape), np.copy(bkg_sub_frame[:,512:1024])
    second_amp_roeba[:,::2] = np.nanmedian(second_amp_data[:,::2])    # Computing median of all even columns
    second_amp_roeba[:,1::2] = np.nanmedian(second_amp_data[:,1::2])  # Computing median of all odd columns
    data_frame[:,512:1024] = data_frame[:,512:1024] - second_amp_roeba
    if fast_read:
        ### Fast-read correction (i.e., along the rows)
        dummy_frame = np.copy(data_frame)
        dummy_frame[mask == 0.] = np.nan
        if row_by_row:
            data_frame[:,512:1024] = data_frame[:,512:1024] - np.nanmedian(dummy_frame[:,512:1024], axis=1)[:,None]
        else:
            data_frame[:,512:1024] = data_frame[:,512:1024] - np.nanmedian(dummy_frame[:,512:1024])

    ## Third amplifier ROEBA
    ### Slow-read correction (i.e., along the columns)
    third_amp_roeba, third_amp_data = np.zeros(bkg_sub_frame[:,1024:1536].shape), np.copy(bkg_sub_frame[:,1024:1536])
    third_amp_roeba[:,::2] = np.nanmedian(third_amp_data[:,::2])   # Computing median of all even columns
    third_amp_roeba[:,1::2] = np.nanmedian(third_amp_data[:,1::2]) # Computing median of all odd columns
    data_frame[:,1024:1536] = data_frame[:,1024:1536] - third_amp_roeba
    if fast_read:
        ### Fast-read correction (i.e., along the rows)
        dummy_frame = np.copy(data_frame)
        dummy_frame[mask == 0.] = np.nan
        if row_by_row:
            data_frame[:,1024:1536] = data_frame[:,1024:1536] - np.nanmedian(dummy_frame[:,1024:1536], axis=1)[:,None]
        else:
            data_frame[:,1024:1536] = data_frame[:,1024:1536] - np.nanmedian(dummy_frame[:,1024:1536])

    ## Fourth amplifier ROEBA
    ### Slow-read correction (i.e., along the columns)
    fourth_amp_roeba, fourth_amp_data = np.zeros(bkg_sub_frame[:,1536:2048].shape), np.copy(bkg_sub_frame[:,1536:2048])
    fourth_amp_roeba[:,::2] = np.nanmedian(fourth_amp_data[:,::2])    # Computing median of all even columns
    fourth_amp_roeba[:,1::2] = np.nanmedian(fourth_amp_data[:,1::2])  # Computing median of all odd columns
    data_frame[:,1536:2048] = data_frame[:,1536:2048] - fourth_amp_roeba
    if fast_read:
        ### Fast-read correction (i.e., along the rows)
        dummy_frame = np.copy(data_frame)
        dummy_frame[mask == 0.] = np.nan
        if row_by_row:
            data_frame[:,1536:2048] = data_frame[:,1536:2048] - np.nanmedian(dummy_frame[:,1536:2048], axis=1)[:,None]
        else:
            data_frame[:,1536:2048] = data_frame[:,1536:2048] - np.nanmedian(dummy_frame[:,1536:2048])

    return data_frame

def computeRMS(data, maxnbins=None, binstep=1, isrmserr=False):
    """Compute the root-mean-squared and standard error for various bin sizes.
    Parameters: This function is taken from the code `Eureka` -- please cite them!!
    ----------
    data : ndarray
        The residuals after fitting.
    maxnbins : int; optional
        The maximum number of bins. Use None to default to 10 points per bin.
    binstep : int; optional
        Bin step size. Defaults to 1.
    isrmserr : bool
        True if return rmserr, else False. Defaults to False.
    Returns
    -------
    rms : ndarray
        The RMS for each bin size.
    stderr : ndarray
        The standard error for each bin size.
    binsz : ndarray
        The different bin sizes.
    rmserr : ndarray; optional
        The uncertainty in the RMS. Only returned if isrmserr==True.
    Notes
    -----
    History:
    - December 29-30, 2021 Taylor Bell
        Moved code to separate file, added documentation.
    """
    data = np.ma.masked_invalid(np.ma.copy(data))
    
    # bin data into multiple bin sizes
    npts = data.size
    if maxnbins is None:
        maxnbins = npts / 10.
    binsz = np.arange(1, maxnbins + binstep, step=binstep, dtype=int)
    nbins = np.zeros(binsz.size, dtype=int)
    rms = np.zeros(binsz.size)
    rmserr = np.zeros(binsz.size)
    for i in range(binsz.size):
        nbins[i] = int(np.floor(data.size / binsz[i]))
        bindata = np.ma.zeros(nbins[i], dtype=float)
        # bin data
        # ADDED INTEGER CONVERSION, mh 01/21/12
        for j in range(nbins[i]):
            bindata[j] = np.ma.mean(data[j * binsz[i]:(j + 1) * binsz[i]])
        # get rms
        rms[i] = np.sqrt(np.ma.mean(bindata ** 2))
        rmserr[i] = rms[i] / np.sqrt(2. * nbins[i])
    # expected for white noise (WINN 2008, PONT 2006)
    stderr = (np.ma.std(data) / np.sqrt(binsz)) * np.sqrt(nbins / (nbins - 1.))
    if isrmserr is True:
        return rms, stderr, binsz, rmserr
    else:
        return rms, stderr, binsz