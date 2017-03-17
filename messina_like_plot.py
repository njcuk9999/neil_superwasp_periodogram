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
try:
    from clean_periodogram import clean_periodogram
except ModuleNotFoundError:
    raise Exception("clean_periodogram.py needed")
try:
    from Neil_clean import neil_clean
except ModuleNotFoundError:
    raise Exception(" Neail_clean.py needed")
try:
    from non_clean_periodogram_functions import lombscargle_periodogram
    from non_clean_periodogram_functions import phase_fold
    from non_clean_periodogram_functions import iFAP
    from non_clean_periodogram_functions import fap_montecarlo
except ModuleNotFoundError:
    raise Exception("Program requires 'non_clean_periodogram_functions.py'")


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

# periodogram constants
SAMPLES_PER_PEAK = 4
NYQUIST_FACTOR = 100


TIME_CONST = 2453800

MESSINA_PERIOD = 3.237


# =============================================================================
# Define functions
# =============================================================================
def messina_plot(time, data, edata, name, lfreq, lpower, mlfreq, mlpower,
                 cfreq, cpower, mcfreq, mcpower, tfold, tfit, dfit,
                 nyquist_factor, samples_per_peak):

    limits = [0.1, 100]
    levels = np.array([0.01])
    faplevels = iFAP(levels, len(time), samples_per_peak, nyquist_factor)


    plt.close()
    fig = plt.figure()
    fig.set_size_inches(16, 12)
    shape = (2, 3)
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
    ltime = 1.0/lfreq
    lmask = (ltime > limits[0]) & (ltime < limits[1])
    normed_lpower = lpower[lmask]
    frame2.plot(ltime[lmask], normed_lpower, lw=0.5, zorder=2, color='k')
    frame2.set_xlabel('time (d)')
    frame2.set_ylabel('$P_N$')
    frame2.set_title('Lomb-Scagle')
    frame2.set_xscale('log')
    frame2.set_xlim(0.1, 100)
    xmin, xmax, ymin, ymax = frame2.axis()
    frame2.vlines(MESSINA_PERIOD, ymin, ymax,
                  colors='b', linestyles='solid', zorder=1, alpha=0.25)
    frame2.set_ylim(ymin, ymax)
    frame2.hlines(faplevels, xmin, xmax,
                  colors='r', linestyles='dashed', zorder=1, alpha=0.5)

    mltime = np.array(1.0/mlfreq)
    mlmask = (mltime > limits[0]) & (mltime < limits[1])
    # ##########################################################################
    # This is a total HACK and has no real justification
    normed_mlpower = abs(mlpower[mlmask] - np.mean(mlpower[mlmask]))
    normed_mlpower = np.max(normed_lpower)*normed_mlpower/np.max(normed_mlpower)
    # ##########################################################################
    frame2.plot(mltime[mlmask], normed_mlpower, linestyle='--',
                lw=0.5, zorder=3, color='r')

    # frame 3: Clean
    ctime = np.array(1.0/cfreq)
    cmask = (ctime > limits[0]) & (ctime < limits[1])
    normed_cpower =cpower[cmask]
    frame3.plot(ctime[cmask], cpower[cmask], lw=0.5, zorder=2, color='k')
    frame3.set_xlabel('time (d)')
    frame3.set_ylabel('$P_N$')
    frame3.set_title('Clean')
    frame3.set_xscale('log')
    frame3.set_xlim(0.1, 100)
    xmin, xmax, ymin, ymax = frame3.axis()
    frame3.vlines(MESSINA_PERIOD, ymin, ymax,
                  colors='b', linestyles='solid', zorder=1, alpha=0.25)
    frame3.set_ylim(ymin, ymax)
    frame2.hlines(faplevels, xmin, xmax,
                  colors='r', linestyles='dashed', zorder=1, alpha=0.5)

    mctime = np.array(1.0/mcfreq)
    mcmask = (mctime > limits[0]) & (mctime < limits[1])
    # ##########################################################################
    # This is a total HACK and has no real justification
    normed_mcpower = abs(mcpower[mcmask] - np.mean(mcpower[mcmask]))
    normed_mcpower = np.max(normed_cpower)*normed_mcpower/np.max(normed_mcpower)
    # ##########################################################################
    frame2.plot(mctime[mcmask], normed_mcpower, linestyle='--',
                lw=0.5, zorder=3, color='r')

    # frame 4: Phase folded lightcurve
    frame4.errorbar(tfold, data, yerr=edata, linestyle='None',
                    marker='o', ms=4, color='k')
    frame4.plot(tfit, dfit, color='r')
    frame4.set_xlabel('rotation phase')
    frame4.set_ylabel('WASP V mag')
    frame4.set_ylim(*frame4.get_ylim()[::-1])
    frame4.set_title('Folded on Messina+2016 period')

    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    plt.show()
    plt.close()


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
    time_arr -= TIME_CONST
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
    ckwargs = dict(freqs=None, log=True, full=True, maxsize=10000,
                   fmax=NYQUIST_FACTOR, ppb=SAMPLES_PER_PEAK)
    freqs1, wfn, dft, cdft = clean_periodogram(time_arr, data_arr, **ckwargs)
    # cdft is the amplitudes power = DFT(x) * conj(DFT(x))
    cpower = np.array(cdft*np.conjugate(cdft))
    freqs1a = freqs1[0: len(cpower)]
    # -------------------------------------------------------------------------
    # Run Monte Carlo lombscargle periodogram
    fargs = [time_arr, data_arr]
    fkwargs = dict(freqs=None, log=True, full=True, maxsize=10000,
                   fmax=NYQUIST_FACTOR, ppb=SAMPLES_PER_PEAK)
    lmkwargs = dict(N=100, log=True, nyquist_factor=NYQUIST_FACTOR,
                    samples_per_peak=SAMPLES_PER_PEAK)
    freqs1_mc, cdft_mc, _, _ = fap_montecarlo(clean_periodogram, fargs,
                                              fkwargs, **lmkwargs)
    # cdft is the amplitudes power = DFT(x) * conj(DFT(x))
    cpower_mc = np.array(cdft_mc*np.conjugate(cdft_mc))
    freqs1a_mc = freqs1_mc[0: len(cpower_mc)]
    # -------------------------------------------------------------------------
    # Run lombscargle periodogram
    lkwargs = dict(freqs=None, nyquist_factor=NYQUIST_FACTOR,
                   samples_per_peak=SAMPLES_PER_PEAK)
    freqs2, lpower = lombscargle_periodogram(time_arr, data_arr, edata_arr,
                                              **lkwargs)
    # -------------------------------------------------------------------------
    # Run Monte Carlo lombscargle periodogram
    fargs = [time_arr, data_arr, edata_arr]
    fkwargs = dict(freqs=None)
    lmkwargs = dict(N=100, log=True, nyquist_factor=NYQUIST_FACTOR,
                    samples_per_peak=SAMPLES_PER_PEAK)
    freqs2_mc, lpower_mc, _, _ = fap_montecarlo(lombscargle_periodogram, fargs,
                                           fkwargs, **lmkwargs)
    # -------------------------------------------------------------------------
    # phase fold on Messina period
    timefold, timefit, datafit = phase_fold(time_arr, data_arr, MESSINA_PERIOD)
    # -------------------------------------------------------------------------
    # plot messina plot
    targetname = '{0}  Messina+2016 period = {1}'.format(SID, MESSINA_PERIOD)
    messina_plot(time_arr, data_arr, edata_arr, targetname, freqs2, lpower,
                 freqs2_mc, lpower_mc, freqs1a, cpower, freqs1a_mc, cpower_mc,
                 timefold, timefit, datafit, NYQUIST_FACTOR, SAMPLES_PER_PEAK)


# =============================================================================
# End of code
# =============================================================================
