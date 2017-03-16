#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 16/03/17 at 11:59 AM

@author: neil

Program description here

Version 0.0.0
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.stats import LombScargle
from astropy.table import Table
from astropy import units as u
from tqdm import tqdm
import sys
sys.path.append('../')
try:
    from clean_periodogram import clean_periodogram, lombscargle_periodogram
except ModuleNotFoundError:
    raise Exception("clean_periodogram.py needed")
try:
    from Neil_clean import neil_clean
except ModuleNotFoundError:
    raise Exception(" Neail_clean.py needed")


# =============================================================================
# Define variables
# =============================================================================
# set file paths
WORKSPACE = '/Astro/Projects/RayPaul_Work/SuperWASP/'
DPATH = WORKSPACE + '/Data/Elodie/'
# -----------------------------------------------------------------------------
SID = 'BPC_46A'
# -----------------------------------------------------------------------------
# Column info
TIMECOL = 'HJD'
DATACOL = 'MAG2'
EDATACOL = 'MAG2_ERR'
# -----------------------------------------------------------------------------
# whether to bin data
BINDATA = True
BINSIZE = 0.1
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


TIME_CONST = 2453800

MESSINA_PERIOD = 3.237

# =============================================================================
# Define functions
# =============================================================================
def messina_plot(time, data, edata, name, lfreq, lpower, cfreq, camp,
                 tfold, tfit, dfit):

    plt.close()
    fig = plt.figure()
    fig.set_size_inches(16, 12)
    shape = (2,3)
    frame1 = plt.subplot2grid(shape, (0, 0), rowspan=1, colspan=1)
    frame2 = plt.subplot2grid(shape, (0, 1), rowspan=1, colspan=1)
    frame3 = plt.subplot2grid(shape, (0, 2), rowspan=1, colspan=1)
    frame4 = plt.subplot2grid(shape, (1, 0), rowspan=1, colspan=3)

    # frame 1: raw data
    frame1.scatter(time, data, s=5, color='k')
    if TIME_CONST is None:
        frame1.set_xlabel('time (HJD)')
    else:
        frame1.set_xlabel('time (HJD - {0})'.format(TIME_CONST))
    frame1.set_ylabel('WASP V mag')
    frame1.set_title(name)
    frame1.set_ylim(*frame1.get_ylim()[::-1])

    # frame 2: Lomb-Scargle
    frame2.plot(1.0/lfreq, lpower, lw=0.5, zorder=2)
    frame2.set_xlabel('time (d)')
    frame2.set_ylabel('$P_N$')
    frame2.set_title('Lomb-Scagle')
    frame2.set_xscale('log')
    frame2.set_xlim(0.01, 100)
    xmin, xmax, ymin, ymax = frame2.axis()
    frame2.vlines(MESSINA_PERIOD, ymin, ymax,
                  colors='r', linestyles='dashed', zorder=1, alpha=0.5)
    frame2.set_ylim(ymin, ymax)

    # frame 3: Clean
    frame3.plot(1.0/cfreq, camp, lw=0.5, zorder=2)
    frame3.set_xlabel('time (d)')
    frame3.set_ylabel('Amplitude')
    frame3.set_title('Clean')
    frame3.set_xscale('log')
    frame3.set_xlim(0.01, 100)
    xmin, xmax, ymin, ymax = frame3.axis()
    frame3.vlines(MESSINA_PERIOD, ymin, ymax,
                  colors='r', linestyles='dashed', zorder=1, alpha=0.5)
    frame3.set_ylim(ymin, ymax)

    # frame 4: Phase folded lightcurve
    frame4.errorbar(tfold, data, yerr=edata, linestyle='None',
                    marker='o', ms=4, color='k')
    frame4.plot(tfit, dfit, color='r')
    frame4.set_xlabel('rotation phase')
    frame4.set_ylabel('WASP V mag')
    frame4.set_ylim(*frame4.get_ylim()[::-1])
    frame4.set_title('Folded on Messina+2016 period')

    plt.show()
    plt.close()


def phase_fold(time, data, period):
    # fold the xdata at given period
    timefold = (time / period) % 1
    # commute the lomb-scargle model at given period
    tfit = np.linspace(0.0, time.max(), 1000)
    yfit = LombScargle(time, data).model(tfit, 1.0/period)
    tfitfold = (tfit / period) % 1
    fsort = np.argsort(tfitfold)
    return timefold, tfitfold[fsort], yfit[fsort]


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
    time_arr = time_arr - TIME_CONST
    data_arr = np.array(lightcurve[DATACOL])
    edata_arr = np.array(lightcurve[EDATACOL])
    # ----------------------------------------------------------------------
    # clean data
    nkwargs = dict(bindata=BINDATA, binsize=BINSIZE, sigmaclip=SIGMACLIP,
                   sigma=SIGMA, size=SIZE, errorclip=ERRORCLIP,
                   percentage=PERCENTAGE)
    time_arr, data_arr, edata_arr = neil_clean(time_arr, data_arr, edata_arr,
                                               **nkwargs)
    # -------------------------------------------------------------------------
    # Run clean periodogram
    ckwargs = dict(freqs=None, log=True, full=True, maxsize=10000)
    freqs1, wfn, dft, cdft = clean_periodogram(time_arr, data_arr, **ckwargs)
    camp1 = 2.0*abs(cdft)
    freqs1a = freqs1[0: len(camp1)]
    # -------------------------------------------------------------------------
    # Run lombscargle periodogram
    freqs2, power = lombscargle_periodogram(time_arr, data_arr, freqs=None)
    # -------------------------------------------------------------------------
    # phase fold on Messina period
    timefold, timefit, datafit = phase_fold(time_arr, data_arr, MESSINA_PERIOD)
    # -------------------------------------------------------------------------
    # plot messina plot
    targetname = '{0}  Messina+2016 period = {1}'.format(SID, MESSINA_PERIOD)
    messina_plot(time_arr, data_arr, edata_arr, targetname, freqs2, power,
                 freqs1a, camp1, timefold, timefit, datafit)


# =============================================================================
# End of code
# =============================================================================
