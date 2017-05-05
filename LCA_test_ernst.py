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
from astropy.table import Table
import matplotlib.pyplot as plt
import sys
import os
import mpmath
from tqdm import tqdm as wrap
try:
    import periodogram_functions2 as pf2
except ImportError:
    raise Exception("Program requires 'periodogram_functions.py'")


# =============================================================================
# Define variables
# =============================================================================
# type of run
TYPE = "Normal"
# TYPE = "DATABASE"
# TYPE = "Elodie"
# -----------------------------------------------------------------------------
# Deal with choosing a target and data paths
WORKSPACE = "/Astro/Projects/RayPaul_Work/SuperWASP/"
# individual location values (for types of run)
if TYPE == "Elodie":
    SID = 'GJ1289'
    # SID = 'GJ793'
    TIMECOL = "time"
    DATACOL = "flux"
    EDATACOL = "eflux"
    # for GJ1289
    if SID == 'GJ1289':
        DPATH = WORKSPACE + "Data/test_for_ernst/bl_gj1289.fits"
    elif SID == 'GJ793':
        DPATH = WORKSPACE + "Data/test_for_ernst/bl_gj793.fits"
    else:
        DPATH = None
elif TYPE == "DATABASE":
    SID = "ABD_108A"
    TIMECOL = 'HJD'
    DATACOL = 'MAG2'
    EDATACOL = 'MAG2_ERR'
else:
    # set file paths
    DPATH = WORKSPACE + 'Data/test_for_ernst/'
    DPATH += '1SWASP J192338.19-460631.5.fits'
    PLOTPATH = WORKSPACE + '/Plots/Messina_like_plots_from_exoarchive/'
    # Column info
    TIMECOL = 'HJD'
    DATACOL = 'TAMMAG2'
    EDATACOL = 'TAMMAG2_ERR'
    SID = "1SWASP J192338.19-460631.5"
# -----------------------------------------------------------------------------
# set database settings
HOSTNAME = 'localhost'
USERNAME = 'root'
PASSWORD = '1234'
DATABASE = 'swasp'
TABLE = 'swasp_sep16_tab'
# -----------------------------------------------------------------------------
# whether to show the graph
SHOW = False
# size in inches of the plot
FIGSIZE = (20, 16)
# decide whether to plot nan periods (saves time)
PLOT_NAN_PERIOD = True
# Name the object manually
NAME = SID
# whether to log progress to standard output (print)
LOG = True
# -----------------------------------------------------------------------------
# minimum time period to be sensitive to (5 hours)
TMIN = 5/24.0
# maximum time period to be sensitive to (100 days)
TMAX = 100
# number of samples per peak
SPP = 5
# -----------------------------------------------------------------------------
# random seed for bootstrapping
RANDOM_SEED = 9
# number of bootstraps to perform
N_BS = 500
# Phase offset
OFFSET = (-0.1, 0.1)
# -----------------------------------------------------------------------------
# number of peaks to find
NPEAKS = 5
# number of pixels around a peak to class as same peak
BOXSIZE = 5
# percentage around noise peak to rule out true peak
THRESHOLD = 5.0
# percentile (FAP) to cut peaks at (i.e. any below are not used)
CUTPERCENTILE = pf2.sigma2percentile(1.0)*100
# -----------------------------------------------------------------------------
# minimum number of data points to define a sub region
MINPOINTS = 50   # points
# maximum gap between data points to define a sub region
MAXGAP = 20  # days
# extention for subgroup
EXT = ''
# how to normalise all powers
NORMALISATION = None

# -----------------------------------------------------------------------------
LIMIT_RANGE = False
RANGE_LOW = 3700
RANGE_HIGH = 4000
# -----------------------------------------------------------------------------
TEST = True
TEST_PERIOD = 3.2
# number of days to observe
TEST_TMAX = 800
# number of observations across TMAX to select
TEST_N = 100
# edata uncertainty (amplitude of signal is set to 1)
TEST_SNR = 2


# =============================================================================
# Define functions
# =============================================================================
def get_params():
    cfmts = [tuple, list, np.ndarray]
    # -------------------------------------------------------------------------
    # get constants from code defined above (only select those in uppercase)
    default_param_names = list(globals().keys())
    params, fmts = dict(), dict()
    for name in default_param_names:
        if name.isupper():
            # get name from global
            params[name] = globals()[name]
            # deal with typing
            kind = type(globals()[name])
            # if a list/array need a type for each element
            if kind in cfmts:
                fmts[name] = [kind, type(globals()[name][0])]
            # else just need a type for the object
            else:
                fmts[name] = [kind, kind]
    # -------------------------------------------------------------------------
    # update these values from the command line
    name, value = "", ""
    args = sys.argv

    try:
        # loop around parameters
        for name in list(params.keys()):
            # loop around commandline arguments
            for arg in args:
                # if no equals then not a valid argument
                if "=" not in arg:
                    continue
                # get the argument and its string value
                argument, value = arg.split('=')
                value = value.replace('"', '')
                # if we recognise the argument use it over the default vlue
                if name == argument:
                    # deal with having lists
                    # i.e. need to cast the type for each element
                    if fmts[name][0] in cfmts:
                        params[name] = array_from_string(value, name,
                                                         fmts[name][0],
                                                         fmts[name][1])
                    # need to deal with booleans
                    elif fmts[name][0] == bool:
                        if value == 'False':
                            params[name] = False
                        else:
                            params[name] = True
                    # all Nones must be string format (as we have no way to tell
                    # what they should be)
                    elif isinstance(None, fmts[name][0]):
                        params[name] = value
                    # else cast the string into type defined from defaults
                    else:
                        params[name] = fmts[name][0](value)
    except ValueError:
        e = ["Error: Parameter ", name, value, fmts[name][0]]
        raise ValueError("{0} {1}={2} must be of type {3}".format(*e))
    # return parameters
    return params


def array_from_string(string, name, fmtarray, fmtelement):
    string = string.split('[')[-1].split('(')[-1]
    string = string.split(']')[0].split(')')[0]
    string = string.replace(',', '')
    rawstringarray = string.split()
    try:
        array = [fmtelement(rsa) for rsa in rawstringarray]
    except ValueError:
        e = ["Error: Parameter ", name, string, fmtarray, 'with elements: ',
             fmtelement]
        raise Exception("{0} {1}={2} must be a {3} {4} {5}".format(*e))

    return fmtarray(array)


def load_data(params):
    # loading data
    if TEST:
        params['NAME'] = "TEST P={0}".format(TEST_PERIOD)
        time, data, edata = test_data(show=False)
        # zero time data to nearest thousand (start at 0 in steps of days)
        day0 = np.floor(time.min() / 100) * 100
        time -= day0
        params['DAY0'] = day0
    elif params['TYPE'] == "DATABASE":
        print('\n Loading data...')
        sid = params['SID']
        sql_kwargs = dict(host=params['HOSTNAME'], db=params['DATABASE'],
                          table=params['TABLE'], user=params['USERNAME'],
                          passwd=params['PASSWORD'])

        pdata = pf2.get_lightcurve_data(conn=None, sid=sid, sortcol='HJD',
                                        replace_infs=True, **sql_kwargs)
        time = np.array(pdata[params['TIMECOL']], dtype=float)
        data = np.array(pdata[params['DATACOL']], dtype=float)
        edata = np.array(pdata[params['EDATACOL']], dtype=float)
        params['NAME'] = params['SID']
    else:
        print('\n Loading data...')
        lightcurve = fits.getdata(params['DPATH'], ext=1)
        if params['NAME'] is None:
            params['NAME'] = DPATH.split('/')[-1].split('.')[0]
        # ---------------------------------------------------------------------
        # get columns
        time = np.array(lightcurve[params['TIMECOL']], dtype=float)
        data = np.array(lightcurve[params['DATACOL']], dtype=float)
        edata = np.array(lightcurve[params['EDATACOL']], dtype=float)
    # zero time data to nearest thousand (start at 0 in steps of days)
    day0 = np.floor(time.min() / 100) * 100
    time -= day0
    params['DAY0'] = day0
    # zero data (by median value)
    data = data - np.median(data)

    # -------------------------------------------------------------------------
    if LIMIT_RANGE:
        mask = np.arange(RANGE_LOW, RANGE_HIGH, 1)
        time, data, edata = time[mask], data[mask], edata[mask]
        day0 = np.floor(time.min() / 100) * 100
        time -= day0
        params['DAY0'] = day0
    # -------------------------------------------------------------------------
    return time, data, edata, params


def get_sub_regions(time, params):
    groupmasks = pf2.subregion_mask(time, params['MINPOINTS'], params['MAXGAP'])
    region_names = ['Full']
    for g_it in range(len(groupmasks)):
        region_names.append('R_{0}'.format(g_it + 1))
    return [np.repeat([True], len(time))] + groupmasks, region_names


def update_progress(params):
    if params['LOG']:
        # name of object
        name = '{0}_{1}'.format(params['NAME'], params['EXT'])
        pargs = ['='*50, 'Run for {0}'.format(name)]
        print('{0}\n\t{1}\n{0}'.format(*pargs))


def calculation(inputs, params, mask=None):

    # format time (days from first time)
    time, data, edata, day0 = format_time_days_from_first(*inputs, mask=mask)
    params['day0'] = day0
    # calculate frequency
    freq = pf2.make_frequency_grid(time, fmin=1.0/params['TMAX'],
                                   fmax=1.0/params['TMIN'],
                                   samples_per_peak=params['SPP'])
    # large frequency grid will take a long time or cause a segmentation fault
    nfreq = len(freq)
    if nfreq > 100000:
        raise ValueError("Error: frequency grid too large ({0})".format(nfreq))
    # results
    results = dict()
    # make combinations of nf, ssp and df
    if params['LOG']:
        print('\n Calculating lombscargle...')
    # kwargs = dict(fit_mean=True, fmin=1/TMAX, fmax=1/TMIN, samples_per_peak=SPP)
    kwargs = dict(fit_mean=True, freq=freq)
    lsfreq, lspower, ls = pf2.lombscargle(time, data, edata, **kwargs)
    lspower = pf2.normalise(lspower, NORMALISATION)
    results['lsfreq'] = lsfreq
    results['lspower'] = lspower
    # -------------------------------------------------------------------------
    # compute window function
    if params['LOG']:
        print('\n Calculating window function...')
    # kwargs = dict(fmin=1 / TMAX, fmax=1 / TMIN, samples_per_peak=SPP)
    kwargs = dict(freq=freq)
    wffreq, wfpower = pf2.compute_window_function(time, **kwargs)
    wfpower = pf2.normalise(wfpower, NORMALISATION)
    results['wffreq'] = wffreq
    results['wfpower'] = wfpower
    # -------------------------------------------------------------------------
    # compute noise periodogram
    kwargs = dict(n_iterations=params['N_BS'],
                  random_seed=params['RANDOM_SEED'], norm='standard',
                  fit_mean=True, log=params['LOG'])
    msfreq, mspower, _, _ = pf2.ls_noiseperiodogram(time, data, edata, lsfreq,
                                                    **kwargs)
    mspower = pf2.normalise(mspower, NORMALISATION)
    results['msfreq'] = msfreq
    results['mspower'] = mspower
    # -------------------------------------------------------------------------
    # try to calculate true period
    if params['LOG']:
        print('\n Attempting to locate real peaks...')
    lsargs = dict(freq=lsfreq, power=lspower, number=params['NPEAKS'],
                  boxsize=params['BOXSIZE'])
    # bsargs = dict(ppeaks=bsppeak, percentile=params['CUTPERCENTILE'])
    bsargs = None
    msargs = dict(freq=msfreq, power=mspower, number=params['NPEAKS'],
                  boxsize=params['BOXSIZE'], threshold=params['THRESHOLD'])
    presults = pf2.find_period(lsargs, bsargs, msargs)
    results['periods'] = presults[0]
    results['power_periods'] = presults[1]
    results['nperiods'] = presults[2]
    results['noise_power_periods'] = presults[3]
    # -------------------------------------------------------------------------
    # calcuate phase data
    if params['LOG']:
        print('\n Computing phase curve...')
    phase, phasefit, powerfit = pf2.phase_data(ls, time, results['periods'])
    results['phase'] = phase
    results['phasefit'] = phasefit
    results['powerfit'] = powerfit
    # -------------------------------------------------------------------------
    inputs = [time, data, edata]
    # -------------------------------------------------------------------------
    return inputs, results, params


def format_time_days_from_first(time, data, edata, mask=None):

    if mask is None:
        mask = np.repeat([True], len(time))

    # zero time data to nearest thousand (start at 0 in steps of days)
    day0 = np.floor(time[mask].min() / 100) * 100
    day1 = np.ceil(time[mask].max() / 100) * 100
    # there should be no time series longer than 10000 but some data has
    # weird times i.e. 32 days and 2454340 days hence if time series
    # seems longer than 10000 days cut it by the median days
    if (day1 - day0) > 1e5:
        day0 = np.median(time[mask]) - 5000
        day1 = np.median(time[mask]) + 5000
        # make sure we have no days beyond maximum day
        limitmask = (time > day0) & (time < day1)
        time = time[limitmask & mask]
        data = data[limitmask & mask]
        if edata is not None:
            edata = edata[limitmask & mask]
    else:
        time = time[mask]
        data = data[mask]
        edata = edata[mask]
    # zero time data to nearest thousand (start at 0 in steps of days)
    day0 = np.floor(time.min() / 100) * 100
    # time should be time since first observation
    time -= day0

    # return time
    return time, data, edata, day0


def plot_graph(inputs, results, params):
    # get inputs
    time, data, edata = inputs
    # extract variables from results
    period = results['periods']
    # sort out the name (add extention for sub regions)
    name = '{0}_{1}'.format(params['NAME'], params['EXT'])
    # -------------------------------------------------------------------------
    # do not bother plotting if we get a zero period
    if np.isnan(period[0]) and not params['PLOT_NAN_PERIOD']:
        return 0
    # -------------------------------------------------------------------------
    # set up plot
    if params['LOG']:
        print('\n Plotting graph...')
    plt.close()
    plt.style.use('seaborn-whitegrid')
    fig, frames = plt.subplots(2, 2, figsize=(params['FIGSIZE']))
    # -------------------------------------------------------------------------
    # plot raw data
    kwargs = dict(xlabel='Time / days',
                  ylabel='$\Delta$ TAM Magnitude',
                  title='Raw data for {0}'.format(name))
    frames[0][0] = pf2.plot_rawdata(frames[0][0], time, data, edata, **kwargs)
    frames[0][0].set_ylim(*frames[0][0].get_ylim()[::-1])
    # -------------------------------------------------------------------------
    # plot window function
    if 'wffreq' in results and 'wfpower' in results:
        wffreq, wfpower = results['wffreq'], results['wfpower']
        kwargs = dict(title='Window function',
                      ylabel='Lomb-Scargle Power $P_N$')
        frames[0][1] = pf2.plot_periodogram(frames[0][1], 1.0/wffreq, wfpower,
                                            **kwargs)
        frames[0][1].set_xscale('log')
    # -------------------------------------------------------------------------
    # plot periodogram
    if 'lsfreq' in results and 'lspower' in results:
        lsfreq, lspower = results['lsfreq'], results['lspower']
        kwargs = dict(title='Lomb-Scargle Periodogram',
                      ylabel='Lomb-Scargle Power $P_N$',
                      xlabel='Time / days',
                      zorder=1)
        frames[1][0] = pf2.plot_periodogram(frames[1][0], 1.0/lsfreq, lspower,
                                            **kwargs)
    # -------------------------------------------------------------------------
    # add arrow to periodogram
    if 'lsfreq' in results and 'lspower' in results and 'period' in results:
        lsfreq, lspower = results['lsfreq'], results['lspower']
        period =  results['periods']
        kwargs = dict(firstcolor='r', normalcolor='b', zorder=4)
        frames[1][0] = pf2.add_arrows(frames[1][0], period, lspower, **kwargs)
    # -------------------------------------------------------------------------
    # plot MCMC periodogram (noise periodogram)
    if ('lsfreq' in results and 'lspower' in results and
        'msfreq' in results and 'mspower' in results):
        msfreq, mspower = results['msfreq'], results['mspower']
        # mspower = np.max(lspower) * mspower / np.max(mspower)
        kwargs = dict(color='r', xlabel=None, ylabel=None, xlim=None, ylim=None,
                      zorder=2)
        frames[1][0] = pf2.plot_periodogram(frames[1][0], 1.0/msfreq, mspower,
                                            **kwargs)
        frames[1][0].set_xscale('log')
    # -------------------------------------------------------------------------
    # Plot FAP lines
    lsfreq= results['lsfreq']
    pargs1 = dict(color='b', linestyle='--')
    if 'theoryFAPpower' in results:
        tfap_power = results['theoryFAPpower']
        sigmas = list(tfap_power.keys())
        faps = list(tfap_power.values())
        pf2.add_fap_lines_to_periodogram(frames[1][0], sigmas, faps, **pargs1)

    pargs1 = dict(color='g', linestyle='--')
    if 'bsFAPpower' in results:
        bfap_power = results['bsFAPpower']
        bs_power = results['bs_power']
        sigmas = list(bfap_power.keys())
        faps = list(bfap_power.values())
        pf2.add_fap_lines_to_periodogram(frames[1][0], sigmas, faps, **pargs1)

        # kwargs = dict(color='g', xlabel=None, ylabel=None, xlim=None, ylim=None,
        #               zorder=2)
        # frames[1][0] = pf2.plot_periodogram(frames[1][0], 1.0/lsfreq, bs_power,
        #                                     **kwargs)
    pargs1 = dict(color='c', linestyle='--')
    if 'mcFAPpower' in results:
        mfap_power = results['mcFAPpower']
        ms_power = results['ms_power']
        sigmas = list(mfap_power.keys())
        faps = list(mfap_power.values())
        pf2.add_fap_lines_to_periodogram(frames[1][0], sigmas, faps, **pargs1)
        # kwargs = dict(color='c', xlabel=None, ylabel=None, xlim=None, ylim=None,
        #               zorder=2)
        # frames[1][0] = pf2.plot_periodogram(frames[1][0], 1.0 / lsfreq, ms_power,
        #                                     **kwargs)

    # -------------------------------------------------------------------------
    # plot phased periodogram
    if 'phase' in results and 'phasefit' in results and 'powerfit' in results:
        phase = results['phase']
        phasefit, powerfit = results['phasefit'], results['powerfit']
        args = [frames[1][1], phase, data, edata, phasefit, powerfit,
                params['OFFSET']]
        kwargs = dict(title='Phase Curve, period={0:.3f} days'.format(period[0]),
                      ylabel='$\Delta$ TAM Magnitude')
        frames[1][1] = pf2.plot_phased_curve(*args, **kwargs)
        frames[1][1].set_ylim(*frames[1][1].get_ylim()[::-1])
    # -------------------------------------------------------------------------
    # save show close
    plt.subplots_adjust(hspace=0.3)
    plt.show()
    plt.close()


def plot_freq_grid_power(xx, freq, dd):
    plt.close()
    dd= np.array(dd)
    # left right bottom top
    extent = [np.min(1/freq), np.max(1/freq), 0, len(xx)]

    im = plt.imshow(xx[::-1], extent=extent, aspect='auto', vmin=0,
                    vmax=np.max(xx), cmap='plasma')
    plt.scatter(1/dd, range(len(dd)), marker='x', color='g', s=2)
    plt.xlabel('Frequency / days$^{-1}$')
    plt.xlabel('Time / days')
    plt.xscale('log')
    plt.ylabel('Monte carlo iteration')
    plt.grid(False)
    cb = plt.colorbar(im)
    cb.set_label('Power')
    plt.show()
    plt.close()


def comparison_test(inp, res):
    results = res
    time, data, edata = inp

    ff, xx, _ = pf2.lombscargle(time, data, edata, fit_mean=True,
                                freq=freq, norm = 'standard')

    plt.plot(1 / ff, xx, color='k', label='True LS')
    plt.plot(1 / ff, x_arr[0], color='r', label='MC LS')
    plt.legend(loc=0)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time / days')
    plt.ylabel('Power $P_N$')
    plt.title('Lomb-Scargle True vs MCMC test')
    frame = plt.gca()
    # -------------------------------------------------------------------------
    # Plot FAP lines
    pargs1 = dict(color='b', linestyle='--')
    if 'theoryFAPpower' in results:
        tfap_power = res['theoryFAPpower']
        sigmas = list(tfap_power.keys())
        faps = list(tfap_power.values())
        pf2.add_fap_lines_to_periodogram(frame, sigmas, faps, **pargs1)
    pargs1 = dict(color='g', linestyle='--')
    if 'bsFAPpower' in results:
        bfap_power = res['bsFAPpower']
        sigmas = list(bfap_power.keys())
        faps = list(bfap_power.values())
        pf2.add_fap_lines_to_periodogram(frame, sigmas, faps, **pargs1)
    pargs1 = dict(color='c', linestyle='--')
    if 'mcFAPpower' in results:
        mfap_power = res['mcFAPpower']
        sigmas = list(mfap_power.keys())
        faps = list(mfap_power.values())
        pf2.add_fap_lines_to_periodogram(frame, sigmas, faps, **pargs1)

    plt.show()
    plt.close()


# =============================================================================
# Define FAP functions
# =============================================================================

def power_from_prob_theory(inputs, faps=None, percentiles=None):
    time, data, edata = inputs
    if faps is None and percentiles is None:
        raise ValueError("Need to define either faps or percentiles")
    if faps is None:
        faps = 1 - np.array(percentiles)/100.0

    N = len(time)
    faps = np.array(faps)
    Meff = -6.363 + 1.193*N + 0.00098*N**2
    prob = 1 - (1 - faps)**mpmath.mpf(1/Meff)
    power = 1 - (prob)**mpmath.mpf(2/(N-3))

    power[power < (sys.float_info.min * 10)] = 0

    return np.array(power, dtype=float)


def lombscargle_bootstrap(time, data, edata, frequency_grid, n_bootstraps=100,
                          random_seed=None, full=False, norm='standard',
                          fit_mean=True, log=False):
    """
    Perform a bootstrap analysis that resamples the data/edata keeping the
    temporal (time vector) co-ordinates constant

    modified from:
    https://github.com/jakevdp/PracticalLombScargle/blob/master
          /figures/Uncertainty.ipynb

    :param time: numpy array, the time vector

    :param data: numpy array, the data vector

    :param edata: numpy array, the uncertainty vector associated with the data
                  vector

    :param frequency_grid: numpy array, the frequency grid to use on each
                           iteration

    :param n_bootstraps: int, number of bootstraps to perform

    :param random_seed: int, random seem to use

    :param full: boolean, if True return freq at maximum power and maximum
                 powers, else return powers

    :param norm: Lomb-Scargle normalisation
                          (see astropy.stats.LombScargle)

    :param fit_mean: boolean, if True uses a floating mean periodogram
                          (generalised Lomb-Scargle periodogram) else uses
                          standard Lomb-Scargle periodogram

    :param log: boolean, if true displays progress to standard output (console)

    :return:
    """
    rng = np.random.RandomState(random_seed)

    kwargs = dict(fit_mean=fit_mean, freq=frequency_grid, norm=norm)

    def bootstrapped_power():
        # sample with replacement
        resample = rng.randint(0, len(data), len(data))
        # define the Lomb Scargle with resampled data and using frequency_grid
        ff, xx, _ = pf2.lombscargle(time, data[resample], edata[resample],
                                    **kwargs)
        # return frequency at maximum and maximum
        return ff, xx

    # run bootstrap
    f_arr, d_arr, x_arr = [], [], []
    for _ in wrap(range(n_bootstraps)):
        f, x = bootstrapped_power()
        x_arr.append(x)
    # sort
    x_arr = np.array(x_arr)
    argmax = np.argmax(x_arr, axis=1)
    f_arr, d_arr = frequency_grid[argmax], x_arr.flat[argmax]
    # return
    if full:
        median = np.percentile(x_arr, 50, axis=0)
        return frequency_grid, x_arr, f_arr, d_arr
    else:
        return d_arr


def get_gaussian_data(means, stds, samples, rng=None):
    if rng is None:
        rng = np.random.RandomState(None)
    # get n_samples number of gaussians for each time element
    g = []
    for i in range(len(means)):
        g.append(rng.normal(means[i], stds[i], size=samples))
    # transpose so we have 1000 light curves with len(time) length
    return np.array(g).T


def lombscargle_mcmc(time, data, edata, frequency_grid, n_bootstraps=100,
                     random_seed=None, full=False, norm='standard',
                     fit_mean=True, log=False):
    """
    Perform a bootstrap analysis that resamples the data/edata keeping the
    temporal (time vector) co-ordinates constant

    modified from:
    https://github.com/jakevdp/PracticalLombScargle/blob/master
          /figures/Uncertainty.ipynb

    :param time: numpy array, the time vector

    :param data: numpy array, the data vector

    :param edata: numpy array, the uncertainty vector associated with the data
                  vector

    :param frequency_grid: numpy array, the frequency grid to use on each
                           iteration

    :param n_bootstraps: int, number of bootstraps to perform

    :param random_seed: int, random seem to use

    :param full: boolean, if True return freq at maximum power and maximum
                 powers, else return powers

    :param norm: Lomb-Scargle normalisation
                          (see astropy.stats.LombScargle)

    :param fit_mean: boolean, if True uses a floating mean periodogram
                          (generalised Lomb-Scargle periodogram) else uses
                          standard Lomb-Scargle periodogram

    :param log: boolean, if true displays progress to standard output (console)

    :return:
    """
    rng = np.random.RandomState(random_seed)

    kwargs = dict(fit_mean=fit_mean, freq=frequency_grid, norm=norm)
    # generate gaussian arrays
    # Want to sample the white noise i.e. take each data point from a
    # gaussian distribution with mean = 0, std = edata
    zerodata = np.zeros_like(data)
    g = get_gaussian_data(zerodata, abs(edata), n_bootstraps, rng)
    def bootstrapped_power(it):
        # define the Lomb Scargle with resampled data and using frequency_grid
        ff, xx, _ = pf2.lombscargle(time, g[it], edata, **kwargs)
        # return frequency at maximum and maximum
        return ff, xx

    # run bootstrap
    f_arr, d_arr, x_arr = [], [], []
    for it in wrap(range(n_bootstraps)):
        f, x = bootstrapped_power(it)
        x_arr.append(x)
    # sort
    x_arr = np.array(x_arr)
    argmax = np.argmax(x_arr, axis=1)
    f_arr, d_arr = frequency_grid[argmax], x_arr.flat[argmax]
    # return
    if full:
        median = np.percentile(x_arr, 50, axis=0)
        return frequency_grid, x_arr, f_arr, d_arr
    else:
        return d_arr


def power_from_prob_bootstrap(inputs, res, faps=None, percentiles=None):
    time, data, edata = inputs
    freq = res['lsfreq']
    if faps is None and percentiles is None:
        raise ValueError("Need to define either faps or percentiles")
    if faps is None:
        faps = 1 - np.array(percentiles)/100.0

    res1 = lombscargle_bootstrap(time, data, edata, freq, n_bootstraps=N_BS,
                                 full=True, norm='standard', fit_mean=True,
                                 log=True)
    freq, x_arr, f_arr, lres = res1
    # plot_freq_grid_power(x_arr, freq, f_arr)
    return np.percentile(lres, 100 * (1 - faps)), np.max(x_arr, axis=0)
    # return np.percentile(lres, 100*(1 - faps)), np.median(x_arr, axis=0)


def power_from_prob_mcmc(inputs, results, faps=None, percentiles=None):
    time, data, edata = inputs
    freq = results['lsfreq']
    if faps is None and percentiles is None:
        raise ValueError("Need to define either faps or percentiles")
    if faps is None:
        faps = 1 - np.array(percentiles)/100.0

    res1 = lombscargle_mcmc(time, data, edata, freq, n_bootstraps=N_BS,
                            full=True, norm='standard', fit_mean=True,
                            log=True)
    freq, x_arr, f_arr, lres = res1
    # plot_freq_grid_power(x_arr, freq, f_arr)
    return np.percentile(lres, 100 * (1 - faps)), np.max(x_arr, axis=0)
    # return np.percentile(lres, 100*(1 - faps)), np.median(x_arr, axis=0)


def test_data(show=True):
    period = TEST_PERIOD
    tmax = TEST_TMAX
    npoints = TEST_N
    noise = TEST_SNR

    time, data, edata = pf2.create_data(npoints, timeamp=tmax,
                                        signal_to_noise=noise, period=period,
                                        random_state=9)
    if show:
        plt.close()
        plt.scatter(time, data)
        plt.show()
        plt.close()

    return time, data, edata


# =============================================================================
# Start of code
# =============================================================================
# Main code here
# noinspection PyUnboundLocalVariable
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    pp = get_params()
    # -------------------------------------------------------------------------
    # Load data
    time_arr, data_arr, edata_arr, pp = load_data(pp)
    # -------------------------------------------------------------------------
    # define mask and name from
    m, pp['EXT'] = None, "full"
    # -------------------------------------------------------------------------
    # print progress if logging on
    update_progress(pp)
    # -------------------------------------------------------------------------
    # LS/noise/phase Calculation
    inp = time_arr, data_arr, edata_arr
    inp, res, pp = calculation(inp, pp, m)
    # -------------------------------------------------------------------------
    # FAP Calculation
    print('\n Verbose False Alarm Probability calculations...')
    sigmas = [1.0, 2.0, 3.0]
    # get percentiles for sigmas
    percentiles = np.array(pf2.sigma2percentile(sigmas) * 100)
    # get false alarm probability power FROM THEORY
    theory_fap_power = power_from_prob_theory(inp, percentiles=percentiles)
    # get false alarm probability power FROM BOOTSTRAP
    print('\n\t Bootstrap FAP (randomising magnitudes)')
    bs_fap_power, bs_power = power_from_prob_bootstrap(inp, res,
                                                      percentiles=percentiles)
    # get false alarm probability power FROM MONTE CARLO
    print('\n\t MCMC FAP (Gaussian dist: mean=1 std=uncertainties)')
    mc_fap_power, ms_power = power_from_prob_mcmc(inp, res,
                                                  percentiles=percentiles)
    res['bs_power'] = bs_power
    res['ms_power'] = ms_power
    # loop around sigmas
    res['theoryFAPpower'] = dict()
    res['bsFAPpower'] = dict()
    res['mcFAPpower'] = dict()
    print('\nNumber of elements in time vector: {0}'.format(len(inp[0])))
    print('\nNumber of elements in freq grid: {0}'.format(len(res['lsfreq'])))
    for s, sigma in enumerate(sigmas):
        # print statements to compare values
        print('Sigma = {0} Percentile = {1:4f}'.format(sigma, percentiles[s]))
        print('BLUE: FAP theory power = {0:4f}'.format(theory_fap_power[s]))
        print('GREEN: FAP bootstrap power = {0:4f}'.format(bs_fap_power[s]))
        print('CYAN: FAP monte carlo power = {0:4f}'.format(mc_fap_power[s]))
        print('\n')
        # save results to results dict
        res['theoryFAPpower'][sigma] = theory_fap_power[s]
        res['bsFAPpower'][sigma] = bs_fap_power[s]
        res['mcFAPpower'][sigma] = mc_fap_power[s]

    # -------------------------------------------------------------------------
    # plotting
    plot_graph(inp, res, pp)



# =============================================================================
# End of code
# =============================================================================
