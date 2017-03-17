#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/03/17 at 4:55 PM

@author: neil

Program description here

Version 0.0.0
"""

import numpy as np
from astropy.io import fits
from collections import OrderedDict
try:
    from non_clean_periodogram_functions import bin_data
    from non_clean_periodogram_functions import quantile_1D
    from non_clean_periodogram_functions import save_to_file
    from non_clean_periodogram_functions import __tqdmlog__
except ModuleNotFoundError:
    raise Exception("Program requires 'non_clean_periodogram_functions.py'")

# =============================================================================
# Define variables
# =============================================================================
# set file paths
WORKSPACE = '/Astro/Projects/RayPaul_Work/SuperWASP/'
DPATH = WORKSPACE + '/Data/Elodie/'
# -----------------------------------------------------------------------------
# SID = 'ARG_54'
SID = 'BPC_46A'
# -----------------------------------------------------------------------------
# Column info
TIMECOL = 'HJD'
DATACOL = 'MAG2'
EDATACOL = 'MAG2_ERR'
# -----------------------------------------------------------------------------
# whether to bin data
BINDATA = True
BINSIZE = 0.05
# -----------------------------------------------------------------------------
# sigma clipping options
SIGMACLIP = True
# size in pixels (actual size will be median separation between points * SIZE)
SIZE = 100
# sigma of the clip i.e. median(DATACOL) + sigma*std(DATACOL)
SIGMA = 2.0
# whether to use a weighted median based on uncertainties
WEIGHTED = True
# -----------------------------------------------------------------------------
# uncertainty filter
ERRORCLIP = True
PERCENTAGE = 0.5


# =============================================================================
# Define functions
# =============================================================================
def sigma_clip(y=None, x=None, ey=None, sigma=3.0, boxsize=10,
               weighted=False, log=False):
    """
    Performs a sigma clip on all y points (weighted if requested)
    uses a box of size median(dx)*boxsize and a clip of:
       median - sigma*stdev < data < median + sigma*stdev

    where median and stdev are weighted if "weighted"=True

    :param y: numpy array, array to be sigma clipped

    :param x: numpy array or None, if not none used for non uniform data
              where dx != 1, if x is None dx = 1

    :param ey: numpy array, uncertainties associated with y (required for
               weighted median)

    :param sigma: float, how many sigmas away from the median to perform the
                  sigma clip to

    :param boxsize: int, number of pixels for size of box (total size of the
                    box is boxsize * dx (centered on the y-element in question

    :param weighted: boolean, if True and ey is not None, performs a weighted
                     median (and weighted standard deviation) based on
                     quantile_1D

                     i.e. standard deviation = mean(q[84]-q[50], q[50]-q[16])
                          median = q[50]

    :param log: boolean, if True print progress messages to screen

    :return good: numpy array of booleans, mask of the points that are kept
                  by the sigma clipping
    """
    # if x is none use a index grid
    if x is None:
        x = np.arange(0, len(y), 1.0)
    # work out average pixel separation (for box size)
    dx = np.median(x[1:] - x[:-1])
    # loop around each row
    good = np.ones_like(x, dtype=bool)
    for row in __tqdmlog__(range(len(x)), log):
        # need box that is boxsize * dx
        xbox = boxsize // 2 * dx
        # mask for points inside box
        mask = (x > (x[row] - xbox)) & (x < (x[row] + xbox))
        # if use weighted median
        if weighted and (ey is not None):
            weights = 1.0 / ey[mask] ** 2
            median = quantile_1D(y[mask], weights, 0.50)
            upper = quantile_1D(y[mask], weights, 0.682689492137/2.0 + 0.5)
            lower = quantile_1D(y[mask], weights, 0.5 - 0.682689492137/2.0)
            onestd = np.mean([upper - median, median-lower])
        # else use median and have no weighted error
        else:
            median = np.median(y[mask])
            onestd = np.std(y[mask])
        good[row] &= (y[row] > (median - sigma*onestd))
        good[row] &= (y[row] < (median + sigma*onestd))
    # return match of good values (True where good)
    return good


def uncertanty_clip(y, ey, percent=1.0):
    """
    Performs a uncertainty cut based on the the relative percentage uncertainty
    i.e. ey/y < percent/100.0

    :param y: numpy array, array to be sigma clipped

    :param ey: numpy array, uncertainties associated with y

    :param percent: float, maximum allowed percentage uncertainty

   :return good: numpy array of booleans, mask of the points that are kept
                 by the uncertainty filter
    """
    good = ey/y < (percent/100.0)
    return good


def neil_clean(time, data, edata, **kwargs):
    """

    :param time:
    :param data:
    :param edata:
    :param kwargs:
    :return:
    """
    bindata = kwargs.get('bindata', BINDATA)
    binsize = kwargs.get('binsize', BINSIZE)
    sigmaclip = kwargs.get('sigmaclip', SIGMACLIP)
    sigma = kwargs.get('sigma', SIGMA)
    size = kwargs.get('size', SIZE)
    errorclip = kwargs.get('errorclip', ERRORCLIP)
    percentage = kwargs.get('percentage', PERCENTAGE)

    # Bin data
    if bindata:
        bkwargs = dict(binsize=binsize, log=True)
        if edata is None:
            res = bin_data(time, data, **bkwargs)
        else:
            res = bin_data(time, data, edata, **bkwargs)
        time, data, edata = res
    # ----------------------------------------------------------------------
    # Sigma Clip
    if sigmaclip:
        gmask1 = sigma_clip(data, x=time, sigma=sigma, boxsize=size,
                            weighted=True)
    else:
        gmask1 = np.ones_like(data)
    # ----------------------------------------------------------------------
    # Uncertainty clip
    if errorclip:
        gmask2 = uncertanty_clip(data, edata, percent=percentage)
    else:
        gmask2 = np.ones_like(data)
    # ----------------------------------------------------------------------
    # combine data masks
    gmask = gmask1 & gmask2
    time = time[gmask]
    data, edata = data[gmask], edata[gmask]
    return time, data, edata


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # load data
    print("\n Loading data...")
    lightcurve = fits.getdata(DPATH + '{0}_lightcurve.fits'.format(SID))
    # ----------------------------------------------------------------------
    # get columns
    time_arr = np.array(lightcurve[TIMECOL])
    data_arr = np.array(lightcurve[DATACOL])
    edata_arr = np.array(lightcurve[EDATACOL])
    # ----------------------------------------------------------------------
    nkwargs = dict(bindata=BINDATA, binsize=BINSIZE, sigmaclip=SIGMACLIP,
                   sigma=SIGMA, size=SIZE, errorclip=ERRORCLIP,
                   percentage=PERCENTAGE)
    time_arr, data_arr, edata_arr = neil_clean(time_arr, data_arr, edata_arr,
                                               **nkwargs)
    # ----------------------------------------------------------------------
    # push back into dictionary
    pdata = OrderedDict(time=time_arr,
                        mag=data_arr,
                        emag=edata_arr)
    # ---------------------------------------------------------------------
    # Save as fits file
    sargs = [SID, 'binsize={0}'.format(BINSIZE), 'sigma={0}'.format(SIGMA),
             'percent={0}'.format(PERCENTAGE)]
    dname = '{0}_lightcurve2_{1}_{2}_{3}'.format(*sargs)
    print('\n Saving light curve to file...')
    save_to_file(pdata, dname, DPATH, exts=['fits'])


# =============================================================================
# End of code
# =============================================================================
