import numpy as np
import juliet
from astropy.stats import SigmaClip, mad_std
import re

def outlier_removal(tims, flx, flxe, clip=3.5):
    # Let's first mask transits and occultations
    per, T0 = 4.0552941, 2455342.96913
    t14 = 0.75*(2.8032/24)
    phs_t = juliet.utils.get_phases(tims, per, T0)
    phs_e = juliet.utils.get_phases(tims, per, (T0+(per/2)))

    mask = np.where((np.abs(phs_e*per) >= t14)&(np.abs(phs_t*per) >= t14))[0]
    tim7, fl7, fle7 = tims[mask], flx[mask], flxe[mask]

    # Sigma clipping
    sc = SigmaClip(sigma_upper=clip, sigma_lower=clip, stdfunc=mad_std, maxiters=None)
    msk1 = sc(fl7).mask

    tim_outliers = tim7[msk1]

    ## Removing outliers from the data
    msk2 = np.ones(len(tims), dtype=bool)
    for i in range(len(tim_outliers)):
        msk2[np.where(tims == tim_outliers[i])[0]] = False
    return msk2

def outlier_removal_ycen(ycen, clip=3.5):
    sc = SigmaClip(sigma_upper=clip, sigma_lower=clip, stdfunc=mad_std, maxiters=None)
    msk1 = sc(ycen).mask

    loc_out = ycen[msk1]

    msk2 = np.ones(len(ycen), dtype=bool)
    for i in range(len(loc_out)):
        msk2[np.where(ycen == loc_out[i])[0]] = False
    return msk2

#------------------------------------------------------------------------------------------
#-------------------------------Natural Sorting--------------------------------------------
#------------------------------------------------------------------------------------------
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]