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