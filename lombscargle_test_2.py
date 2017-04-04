#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 29/03/17 at 11:40 AM

@author: neil

Program description here

Version 0.0.0
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
try:
    import periodogram_functions2 as pf2
except ModuleNotFoundError:
    raise Exception("Program requires 'periodogram_functions.py'")
from scipy.special import erf, erfinv


# =============================================================================
# Define variables
# =============================================================================
WORKSPACE = "/Astro/Projects/RayPaul_Work/SuperWASP/"
# Deal with choosing a target and data paths
Elodie = False
if Elodie:
    SID = 'GJ1289'
    # SID = 'GJ793'
    TIMECOL = "time"
    DATACOL = "flux"
    EDATACOL = "eflux"
    # for GJ1289
    if SID == 'GJ1289':
        DPATH = WORKSPACE + "Data/Elodie/bl_gj1289.fits"
    elif SID == 'GJ793':
        DPATH = WORKSPACE + "Data/Elodie/bl_gj793.fits"
    else:
        DPATH = None
else:
    # set file paths
    DPATH = WORKSPACE + 'Data/from_exoplanetarchive/'
    DPATH += '1SWASP J192338.19-460631.5.fits'
    PLOTPATH = WORKSPACE + '/Plots/Messina_like_plots_from_exoarchive/'
    # Column info
    TIMECOL = 'HJD'
    DATACOL = 'MAG2'
    EDATACOL = 'MAG2_ERR'
# -----------------------------------------------------------------------------
TEST_RUN = True
TEST_PERIOD = 3.28
DT = None
# -----------------------------------------------------------------------------
# minimum time period to be sensitive to
TMIN = 0.1
# maximum time period to be sensitive to
TMAX = 100
# number of samples per peak
SPP = 10
# random seed for bootstrapping
RANDOM_SEED = 9
# number of bootstraps to perform
N_BS = 100
# Phase offset
OFFSET = (-0.5, 0.5)
# define the FAP percentiles
PERCENTILES = [pf2.sigma2percentile(1)*100,
               pf2.sigma2percentile(2)*100,
               pf2.sigma2percentile(3)*100]
# number of peaks to find
NPEAKS = 5
# number of pixels around a peak to class as same peak
BOXSIZE = 5
# percentage around noise peak to rule out true peak
THRESHOLD = 5.0
# percentile (FAP) to cut peaks at (i.e. any below are not used)
CUTPERCENTILE = pf2.sigma2percentile(1.0)*100


# =============================================================================
# Define functions
# =============================================================================
def plot_graph(time, data, edata, name, wffreq, wfpower, lsfreq, lspower, day0,
               lsperiod, bsppeaks, bsfreq, bspower, msfreq, mspower, phase,
               phasefit, powerfit):
    plt.close()
    plt.style.use('seaborn-whitegrid')
    fig, frames = plt.subplots(2, 2, figsize=(16, 16))
    # -------------------------------------------------------------------------
    # plot raw data
    kwargs = dict(xlabel='Time since {0}/ days'.format(day0),
                  ylabel='Magnitude', title='Raw data for {0}'.format(name))
    frames[0][0] = pf2.plot_rawdata(frames[0][0], time, data, edata, **kwargs)
    frames[0][0].set_ylim(*frames[0][0].get_ylim()[::-1])
    # -------------------------------------------------------------------------
    # plot window function
    kwargs = dict(title='Window function')
    frames[0][1] = pf2.plot_periodogram(frames[0][1], 1.0/wffreq, wfpower,
                                        **kwargs)
    frames[0][1].set_xscale('log')
    # -------------------------------------------------------------------------
    # plot periodogram
    kwargs = dict(title='Lomb-Scargle Periodogram',
                  xlabel='Time since {0}/ days'.format(day0), zorder=1)
    frames[1][0] = pf2.plot_periodogram(frames[1][0], 1.0/lsfreq, lspower,
                                        **kwargs)
    # add arrow to periodogram
    kwargs = dict(firstcolor='r', normalcolor='b', zorder=4)
    frames[1][0] = pf2.add_arrows(frames[1][0], lsperiod, lspower, **kwargs)
    # add FAP lines to periodogram
    kwargs = dict(color='b', zorder=4)
    frames[1][0] = pf2.add_fap_to_periodogram(frames[1][0], bsppeaks,
                                              PERCENTILES, **kwargs)
    # plot bootstrap periodogram (noise periodogram)
    kwargs = dict(color='0.5', xlabel=None, ylabel=None, xlim=None, ylim=None,
                  zorder=0, alpha=0.25)
    frames[1][0] = pf2.plot_periodogram(frames[1][0], 1.0/bsfreq, bspower,
                                        **kwargs)

    # plot MCMC periodogram (noise periodogram)
    kwargs = dict(color='m', xlabel=None, ylabel=None, xlim=None, ylim=None,
                  zorder=2)
    frames[1][0] = pf2.plot_periodogram(frames[1][0], 1.0/msfreq, mspower,
                                        **kwargs)
    frames[1][0].set_xscale('log')
    # -------------------------------------------------------------------------
    # plot phased periodogram
    args = [frames[1][1], phase, data, edata, phasefit, powerfit, OFFSET]
    kwargs = dict(title='Phase Curve, period={0:.3f} days'.format(lsperiod[0]),
                  ylabel='Magnitude')
    frames[1][1] = pf2.plot_phased_curve(*args, **kwargs)
    frames[1][1].set_ylim(*frames[1][1].get_ylim()[::-1])
    # -------------------------------------------------------------------------
    # save show close
    plt.subplots_adjust(hspace=0.3)
    plt.show()
    plt.close()


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    if not TEST_RUN:
        # ---------------------------------------------------------------------
        # loading data
        print('\n Loading data...')
        lightcurve = fits.getdata(DPATH, ext=1)
        name = DPATH.split('/')[-1].split('.')[0]
        # ---------------------------------------------------------------------
        # get columns
        time = np.array(lightcurve[TIMECOL], dtype=float)
        data = np.array(lightcurve[DATACOL], dtype=float)
        edata = np.array(lightcurve[EDATACOL], dtype=float)
        # zero time data to nearest thousand (start at 0 in steps of days)
        day0 = np.floor(time.min()/100)*100
        time -= day0
    # -------------------------------------------------------------------------
    else:
        kwargs = dict(N=30, T=TMAX, period=TEST_PERIOD, dt=DT,
                      signal_to_noise=5,  random_state=583)
        time, data, edata = pf2.create_data(**kwargs)
        name = 'Test data, period={0}'.format(TEST_PERIOD)
        day0 = 0
    # -------------------------------------------------------------------------
    # Calculation
    # -------------------------------------------------------------------------
    # make combinations of nf, ssp and df
    print('\n Calculating lombscargle...')
    kwargs = dict(fit_mean=True, fmin=1/TMAX, fmax=1/TMIN, samples_per_peak=SPP)
    lsfreq, lspower, ls = pf2.lombscargle(time, data, edata, **kwargs)
    lspower = pf2.normalise(lspower)
    # -------------------------------------------------------------------------
    # compute window function
    print('\n Calculating window function...')
    kwargs = dict(fmin=1 / TMAX, fmax=1 / TMIN, samples_per_peak=SPP)
    wffreq, wfpower = pf2.compute_window_function(time, **kwargs)
    wfpower = pf2.normalise(wfpower)
    # -------------------------------------------------------------------------
    # compute bootstrap of lombscargle
    print('\n Computing bootstrap...')
    kwargs = dict(n_bootstraps=N_BS, random_seed=RANDOM_SEED, norm='standard',
                  fit_mean=True, log=True, full=True)
    bsresults = pf2.lombscargle_bootstrap(time, data, edata, lsfreq, **kwargs)
    bsfreq, bspower, bsfpeak, bsppeak,  = bsresults
    bspower = pf2.normalise(bspower)
    bsppeak = pf2.normalise(bsppeak)
    # -------------------------------------------------------------------------
    # compute bootstrap of lombscargle
    print('\n Computing MCMC...')
    kwargs = dict(N_iterations=N_BS, random_seed=RANDOM_SEED, norm='standard',
                  fit_mean=True, log=True)
    msfreq, mspower, _, _ = pf2.ls_montecarlo(time, data, edata, lsfreq,
                                              **kwargs)
    mspower = pf2.normalise(mspower)
    # -------------------------------------------------------------------------
    # try to calculate true period
    print('Attempting to locate real peaks...')
    lsargs = dict(freq=lsfreq, power=lspower, number=NPEAKS, boxsize=BOXSIZE)
    bsargs = dict(ppeaks=bsppeak, percentile=CUTPERCENTILE)
    msargs = dict(freq=msfreq, power=mspower, number=NPEAKS, boxsize=BOXSIZE,
                  threshold=THRESHOLD)
    period = pf2.find_period(lsargs, bsargs, msargs)
    # -------------------------------------------------------------------------
    # calcuate phase data
    print('\n Computing phase curve...')
    phase, phasefit, powerfit = pf2.phase_data(ls, time, period)
    # -------------------------------------------------------------------------
    # plotting
    # -------------------------------------------------------------------------
    print('\n Plotting graph...')
    plotargs = [time, data, edata, name, wffreq, wfpower, lsfreq, lspower, day0,
                period, bsppeak, bsfreq, bspower, msfreq, mspower, phase,
                phasefit, powerfit]
    plot_graph(*plotargs)



# =============================================================================
# End of code
# =============================================================================
