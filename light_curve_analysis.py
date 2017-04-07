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
try:
    import periodogram_functions2 as pf2
except ModuleNotFoundError:
    raise Exception("Program requires 'periodogram_functions.py'")


# =============================================================================
# Define variables
# =============================================================================
# type of run
# TYPE = "Normal"
# TYPE = "Elodie"
# TYPE = "Test"
TYPE = "Database"
# -----------------------------------------------------------------------------
# Deal with choosing a target and data paths
WORKSPACE = "/Astro/Projects/RayPaul_Work/SuperWASP/"
# location of folder to plot files to
PLOTPATH = WORKSPACE + '/Plots/ls_analysis_run/'
# file to save periods to
PERIODPATH = WORKSPACE + '/Data/ls_analysis_run/'
PERIODPATH += 'light_curve_analysis_periods_regions.fits'
# if True and periodpath file exists we will skip entries that exist
SKIP_DONE = True
# Write to file (if false does not save to file)
WRITE_TO_FILE = True
# individual location values (for types of run)
if TYPE == "Elodie":
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
    TEST_RUN, FROM_DATABASE = False, False
elif TYPE == "Test":
    TEST_RUN, FROM_DATABASE = True, False
elif TYPE == "Database":
    FROM_DATABASE = True
    TIMECOL = 'HJD'
    DATACOL = 'MAG2'
    EDATACOL = 'MAG2_ERR'
else:
    # set file paths
    DPATH = WORKSPACE + 'Data/from_exoplanetarchive/'
    DPATH += '1SWASP J192338.19-460631.5.fits'
    PLOTPATH = WORKSPACE + '/Plots/Messina_like_plots_from_exoarchive/'
    # Column info
    TIMECOL = 'HJD'
    DATACOL = 'MAG2'
    EDATACOL = 'MAG2_ERR'
    TEST_RUN, FROM_DATABASE = False, False
# -----------------------------------------------------------------------------
# Set up test parameters
TEST_N = 300
TEST_PERIOD = 3.28
TEST_DT = None
TEST_SNR = 5
# --------------------------------------------------------------------------
# set database settings
HOSTNAME = 'localhost'
USERNAME = 'root'
PASSWORD = '1234'
DATABASE = 'swasp'
TABLE = 'swasp_sep16_tab'
SID = 'BPC_46A'
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
# minimum time period to be sensitive to
TMIN = 0.1
# maximum time period to be sensitive to
TMAX = 100
# number of samples per peak
SPP = 10
# -----------------------------------------------------------------------------
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
    # -----------------------------------------------------------------------
    if params['FROM_DATABASE']:
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
        # zero time data to nearest thousand (start at 0 in steps of days)
        day0 = np.floor(time.min() / 100) * 100
        time -= day0
        params['DAY0'] = day0
        params['NAME'] = params['SID']
    # -----------------------------------------------------------------------
    elif not params['TEST_RUN']:
        # loading data
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
    # -------------------------------------------------------------------------
    else:
        print('\n Loading data...')
        kwargs = dict(N=params['TESTN'], T=params['TMAX'],
                      period=params['TEST_PERIOD'], dt=params['TEST_DT'],
                      signal_to_noise=params['TEST_SNR'], random_state=583)
        time, data, edata = pf2.create_data(**kwargs)
        if params['NAME'] is None:
            params['NAME'] = 'Test_Data_p={0}'.format(params['TEST_PERIOD'])
        params['DAY0'] = 0
    # -----------------------------------------------------------------------
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


def calculation(time, data, edata, params):

    # calculate frequency
    freq = pf2.make_frequency_grid(time, fmin=1.0/TMAX, fmax=1.0/TMIN,
                                   samples_per_peak=SPP)
    # zero data
    data = data - np.median(data)
    # zero time data to nearest thousand (start at 0 in steps of days)
    day0 = np.floor(time.min() / 100) * 100
    time -= day0
    params['DAY0'] = day0
    # results
    results = dict()
    # make combinations of nf, ssp and df
    if params['LOG']:
        print('\n Calculating lombscargle...')
    # kwargs = dict(fit_mean=True, fmin=1/TMAX, fmax=1/TMIN, samples_per_peak=SPP)
    kwargs = dict(fit_mean=True, freq=freq)
    lsfreq, lspower, ls = pf2.lombscargle(time, data, edata, **kwargs)
    lspower = pf2.normalise(lspower)
    results['lsfreq'] = lsfreq
    results['lspower'] = lspower
    # -------------------------------------------------------------------------
    # compute window function
    if params['LOG']:
        print('\n Calculating window function...')
    # kwargs = dict(fmin=1 / TMAX, fmax=1 / TMIN, samples_per_peak=SPP)
    kwargs = dict(freq=freq)
    wffreq, wfpower = pf2.compute_window_function(time, **kwargs)
    wfpower = pf2.normalise(wfpower)
    results['wffreq'] = wffreq
    results['wfpower'] = wfpower
    # -------------------------------------------------------------------------
    # compute bootstrap of lombscargle
    if params['LOG']:
        print('\n Computing bootstrap...')
    kwargs = dict(n_bootstraps=params['N_BS'],
                  random_seed=params['RANDOM_SEED'], norm='standard',
                  fit_mean=True, log=params['LOG'], full=True)
    bsresults = pf2.lombscargle_bootstrap(time, data, edata, lsfreq, **kwargs)
    bsfreq, bspower, bsfpeak, bsppeak,  = bsresults
    bspower = pf2.normalise(bspower)
    bsppeak = pf2.normalise(bsppeak)
    results['bsfreq'] = bsfreq
    results['bspower'] = bspower
    results['bsfpeak'] = bsfpeak
    results['bsppeak'] = bsppeak
    # -------------------------------------------------------------------------
    # compute bootstrap of lombscargle
    if params['LOG']:
        print('\n Computing MCMC...')
    kwargs = dict(n_iterations=params['N_BS'],
                  random_seed=params['RANDOM_SEED'], norm='standard',
                  fit_mean=True, log=params['LOG'])
    msfreq, mspower, _, _ = pf2.ls_montecarlo(time, data, edata, lsfreq,
                                              **kwargs)
    mspower = pf2.normalise(mspower)
    results['msfreq'] = msfreq
    results['mspower'] = mspower
    # -------------------------------------------------------------------------
    # try to calculate true period
    if params['LOG']:
        print('Attempting to locate real peaks...')
    lsargs = dict(freq=lsfreq, power=lspower, number=params['NPEAKS'],
                  boxsize=params['BOXSIZE'])
    bsargs = dict(ppeaks=bsppeak, percentile=params['CUTPERCENTILE'])
    msargs = dict(freq=msfreq, power=mspower, number=params['NPEAKS'],
                  boxsize=params['BOXSIZE'], threshold=params['THRESHOLD'])
    period, periodpower = pf2.find_period(lsargs, bsargs, msargs)
    results['period'] = period
    results['power_at_period'] = periodpower
    # -------------------------------------------------------------------------
    # calculate FAP at period
    if params['LOG']:
        print('Estimating significane of peaks...')
    # significance = pf2.inverse_fap_from_bootstrap(bsppeak, periodpower, dp=3)
    # results['significance'] = significance
    fap = pf2.fap_from_theory(time, periodpower)
    results['false_alaram_prob'] = fap
    results['significance'] = pf2.percentile2sigma(1.0 - fap)
    # -------------------------------------------------------------------------
    # calcuate phase data
    if params['LOG']:
        print('\n Computing phase curve...')
    phase, phasefit, powerfit = pf2.phase_data(ls, time, period)
    results['phase'] = phase
    results['phasefit'] = phasefit
    results['powerfit'] = powerfit
    # -------------------------------------------------------------------------
    return results, params


def plot_graph(time, data, edata, results, params):
    # zero time data to nearest thousand (start at 0 in steps of days)
    time -= params['DAY0']
    # extract variables from results
    wffreq, wfpower = results['wffreq'], results['wfpower']
    lsfreq, lspower = results['lsfreq'], results['lspower']
    period, bsppeaks = results['period'], results['bsppeak']
    bsfreq, bspower = results['bsfreq'], results['bspower']
    msfreq, mspower = results['msfreq'], results['mspower']
    phase = results['phase']
    phasefit, powerfit = results['phasefit'], results['powerfit']
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
    kwargs = dict(xlabel='Time since {0}/ days'.format(params['DAY0']),
                  ylabel='Magnitude',
                  title='Raw data for {0}'.format(name))
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
                  ylabel='Lomb-Scargle Power $P_N/P_{max}$',
                  xlabel='Time since {0}/ days'.format(params['DAY0']),
                  zorder=1)
    frames[1][0] = pf2.plot_periodogram(frames[1][0], 1.0/lsfreq, lspower,
                                        **kwargs)
    # add arrow to periodogram
    kwargs = dict(firstcolor='r', normalcolor='b', zorder=4)
    frames[1][0] = pf2.add_arrows(frames[1][0], period, lspower, **kwargs)
    # add FAP lines to periodogram
    kwargs = dict(color='b', zorder=4)
    frames[1][0] = pf2.add_fap_to_periodogram(frames[1][0], bsppeaks,
                                              params['PERCENTILES'], **kwargs)
    # plot bootstrap periodogram (noise periodogram)
    kwargs = dict(color='0.5', xlabel=None, ylabel=None, xlim=None, ylim=None,
                  zorder=0, alpha=0.25)
    frames[1][0] = pf2.plot_periodogram(frames[1][0], 1.0/bsfreq, bspower,
                                        **kwargs)

    # renormalise the noise periodogram to the lspower
    mspower = np.max(lspower) * mspower / np.max(mspower)
    # plot MCMC periodogram (noise periodogram)
    kwargs = dict(color='r', xlabel=None, ylabel=None, xlim=None, ylim=None,
                  zorder=2)
    frames[1][0] = pf2.plot_periodogram(frames[1][0], 1.0/msfreq, mspower,
                                        **kwargs)
    frames[1][0].set_xscale('log')
    # -------------------------------------------------------------------------
    # plot phased periodogram
    args = [frames[1][1], phase, data, edata, phasefit, powerfit,
            params['OFFSET']]
    kwargs = dict(title='Phase Curve, period={0:.3f} days'.format(period[0]),
                  ylabel='Magnitude')
    frames[1][1] = pf2.plot_phased_curve(*args, **kwargs)
    frames[1][1].set_ylim(*frames[1][1].get_ylim()[::-1])
    # -------------------------------------------------------------------------
    # save show close
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    if params['SHOW']:
        plt.show()
        plt.close()
    else:
        sname = 'LS_analysis_{0}'.format(name)
        plt.savefig(params['PLOTPATH'] + sname + '.png', bbox_inches='tight')
        plt.savefig(params['PLOTPATH'] + sname + '.pdf', bbox_inches='tight')


def save_to_fit(results, params):
    if not params['WRITE_TO_FILE']:
        return 0
    # print update if logging on
    if params['LOG']:
        print('\n Saving to fits file')
    periods = results['period']
    faps = results['false_alaram_prob']
    sigs = results['significance']
    # name of object
    name = '{0}_{1}'.format(params['NAME'], params['EXT'])
    # load data if it exists
    if os.path.exists(params['PERIODPATH']):
        perioddata = Table.read(params['PERIODPATH'])
        # add this data to the table
        row = [name]
        for p_it, period in enumerate(periods):
            row.append(period)
            row.append(faps[p_it])
            row.append(sigs[p_it])
        perioddata.add_row(row)
    # else create a new table and populate it
    else:
        perioddata = Table()
        perioddata['name'] = [name]
        for p_it, period in enumerate(periods):
            perioddata['Peak{0}'.format(p_it+1)] = [period]
            perioddata['FAP{0}'.format(p_it + 1)] = [faps[p_it]]
            perioddata['sig{0}'.format(p_it + 1)] = [sigs[p_it]]
    # finally save the modified/new table over original table
    perioddata.write(params['PERIODPATH'], format='fits', overwrite=True)
    return 1


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
    # find sub regions
    masks, names = get_sub_regions(time_arr, pp)
    # -------------------------------------------------------------------------
    # loop around masks
    # -------------------------------------------------------------------------
    for m_it in range(len(masks)):
        # ---------------------------------------------------------------------
        # define mask and name from
        m, pp['EXT'] = masks[m_it], names[m_it]
        # ---------------------------------------------------------------------
        # print progress if logging on
        update_progress(pp)
        # ---------------------------------------------------------------------
        # Calculation
        res, pp = calculation(time_arr[m], data_arr[m], edata_arr[m], pp)
        # ---------------------------------------------------------------------
        # plotting
        plot_graph(time_arr[m], data_arr[m], edata_arr[m], res, pp)
        # ---------------------------------------------------------------------
        # save periods to file
        save_to_fit(res, pp)
        # input('Enter to continue. Control+C to cancel')


# =============================================================================
# End of code
# =============================================================================
