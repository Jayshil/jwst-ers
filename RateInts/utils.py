import numpy as np
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