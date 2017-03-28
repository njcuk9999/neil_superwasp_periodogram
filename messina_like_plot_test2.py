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
import sys
import os
try:
    from clean_periodogram import clean_periodogram
except ModuleNotFoundError:
    raise Exception("clean_periodogram.py needed")
try:
    from neil_clean import neil_clean
except ModuleNotFoundError:
    raise Exception(" Neail_clean.py needed")
try:
    from periodogram_functions import lombscargle_periodogram
    from periodogram_functions import phase_fold
    from periodogram_functions import iFAP
    from periodogram_functions import fap_montecarlo
    from periodogram_functions import find_y_peaks
    from periodogram_functions import get_lightcurve_data
    from periodogram_functions import relative_cluster
except ModuleNotFoundError:
    raise Exception("Program requires 'periodogram_functions.py'")

# =============================================================================
# Define variables
# =============================================================================
# set file paths
# set file paths
WORKSPACE = '/Astro/Projects/RayPaul_Work/SuperWASP/'
DPATH = WORKSPACE + 'Data/from_exoplanetarchive/'
DPATH += '1SWASP J192338.19-460631.5.fits'
PLOTPATH = WORKSPACE + '/Plots/Messina_like_plots_from_exoarchive/'
# -----------------------------------------------------------------------------
SID = '1SWASP_J192338-460631'
SKIP = False
# -----------------------------------------------------------------------------
# Column info
TIMECOL = 'HJD'
DATACOL = 'MAG2'
EDATACOL = 'MAG2_ERR'
# -----------------------------------------------------------------------------
# whether to bin data
BINDATA = False
BINSIZE = 0.1
# -----------------------------------------------------------------------------
# sigma clipping options
SIGMACLIP = False
# size in pixels (actual size will be median separation between points * SIZE)
SIZE = 100
# sigma of the clip i.e. median(DATACOL) + sigma*std(DATACOL)
SIGMA = 2.0
# whether to use a weighted median based on uncertainties
WEIGHTED = True
# -----------------------------------------------------------------------------
# uncertainty filter
ERRORCLIP = False
PERCENTAGE = 0.5
# -----------------------------------------------------------------------------
# periodogram constants
SAMPLES_PER_PEAK = 4
NYQUIST_FACTOR = 100
FAP_LEVELS = [0.01]
PEAK_RANGE = [0.1, 100.0]
PEAK_THRESHOLD_PERCENTAGE = 1.0
# -----------------------------------------------------------------------------
# Time to subtract off the time vector
TIME_CONST = 2453800
# Period from messina
MESSINA_PERIOD = 3.237
# -----------------------------------------------------------------------------
# database variables
HOSTNAME = 'localhost'
USERNAME = 'root'
PASSWORD = '1234'
DATABASE = 'swasp'
TABLE = 'swasp_sep16_tab'


# =============================================================================
# Define functions
# =============================================================================
def load_params(**kwargs):

    args = sys.argv
    keys = ['DPATH', 'PLOTPATH', 'SID', 'TIMECOL', 'DATACOL', 'EDATACOL',
            'BINDATA', 'BINSIZE', 'SIGMACLIP', 'SIZE', 'SIGMA', 'WEIGHTED',
            'ERRORCLIP', 'PERCENTAGE', 'SAMPLES_PER_PEAK', 'NYQUIST_FACTOR',
            'FAP_LEVELS', 'PEAK_RANGE', 'PEAK_THRESHOLD_PERCENTAGE',
            'TIME_CONST', 'MESSINA_PERIOD', 'SKIP']
    formats = [str, str, str, str, str, str, bool, float, bool, int, float,
               bool, bool, float, int, int, list, list, float, float, float,
               bool]
    listfmts = dict(fap_levels=float, peak_range=float)
    dvalues = [DPATH, PLOTPATH, SID, TIMECOL, DATACOL, EDATACOL, BINDATA,
               BINSIZE, SIGMACLIP, SIZE, SIGMA, WEIGHTED, ERRORCLIP,
               PERCENTAGE, SAMPLES_PER_PEAK, NYQUIST_FACTOR, FAP_LEVELS,
               PEAK_RANGE, PEAK_THRESHOLD_PERCENTAGE, TIME_CONST,
               MESSINA_PERIOD, SKIP]
    dparams = dict(zip(keys, dvalues))
    params = dict()
    # set default from command line or python constants
    for kit, key in enumerate(keys):
        fmt = formats[kit]
        # look for args from command line
        for arg in args:
            # look for key in commandline argument
            if key == arg.split('=')[0]:
                value = arg.split(key + '=')[-1]
                value = value.replace("'", '').replace('"', '')
                # if format type is list need to construct list from string
                if fmt == list:
                    values = value[1:-1].split(',')
                    fvalues = []
                    for value in values:
                        nvalue = format_error(key, value, listfmts[key.lower()])
                        fvalues.append(nvalue)
                    params[key] =fvalues
                # else need to add to string
                else:
                    params[key] = format_error(key, value, fmt)
        # else populate from default values (define in python)
        if key not in params:
            params[key] = dparams[key]
    # override params with function keyword arguments
    for key in kwargs:
        params[key] = kwargs[key]
    # return params
    return params


def format_error(key, value, fmt):
    try:
        newvalue = fmt(value)
    except ValueError:
        eargs = [key, value, str(fmt)]
        emsg = 'Command line input {0}={1} is not a valid {2}'.format(*eargs)
        raise ValueError(emsg)
    return newvalue


def messina_plot(time, data, edata, name, lfreq, lpower, mlfreq, mlpower,
                 cfreq, cpower, mcfreq, mcpower, tfold, tfit, dfit, lx, ly,
                 mlx, mly, cx, cy, mcx, mcy, glpeaks, **kwargs):

    # sort out keyword arguments
    nyquist_factor = kwargs.get('NYQUIST_FACTOR', NYQUIST_FACTOR)
    samples_per_peak = kwargs.get('SAMPLES_PER_PEAK', SAMPLES_PER_PEAK)
    limits = kwargs.get('PEAK_RANGE', PEAK_RANGE)
    fap_levels = kwargs.get('FAP_LEVELS', FAP_LEVELS)
    timeconst = kwargs.get('TIME_CONST', TIME_CONST)
    percentage = kwargs.get('PEAK_THRESHOLD_PERCENTAGE', PERCENTAGE)
    savepath = kwargs.get('PLOTPATH', PLOTPATH)
    show = kwargs.get('show', False)

    # -------------------------------------------------------------------------
    # set up
    # -------------------------------------------------------------------------
    nperiods = len(tfold)
    plt.close()
    fig = plt.figure()
    fig.set_size_inches(20, 2*20/3.0 + nperiods*20/3.0)
    shape = (2 + nperiods, 3)
    frame1 = plt.subplot2grid(shape, (0, 0), rowspan=1, colspan=1)
    frame2 = plt.subplot2grid(shape, (0, 1), rowspan=1, colspan=1)
    frame3 = plt.subplot2grid(shape, (0, 2), rowspan=1, colspan=1)
    frame4 = plt.subplot2grid(shape, (1, 0), rowspan=1, colspan=3)
    frames = []
    for nperiod in range(nperiods):
        frame = plt.subplot2grid(shape, (nperiod+2, 0), rowspan=1, colspan=3)
        frames.append(frame)
    levels = np.array(fap_levels)
    faplevels = iFAP(levels, len(time), samples_per_peak, nyquist_factor)

    # -------------------------------------------------------------------------
    # frame 1: raw data
    # -------------------------------------------------------------------------
    frame1.scatter(time, data, s=5, color='k')
    if timeconst is None:
        frame1.set_xlabel('time (HJD)')
    else:
        frame1.set_xlabel('time (HJD - {0})'.format(timeconst))
    frame1.set_ylabel('WASP V mag')
    frame1.set_title(name)
    frame1.set_ylim(*frame1.get_ylim()[::-1])

    # -------------------------------------------------------------------------
    # frame 2: Lomb-Scargle
    # -------------------------------------------------------------------------
    ltime = 1.0 / lfreq
    lmask = (ltime > limits[0]) & (ltime < limits[1])
    normed_lpower = lpower[lmask]
    frame2.plot(ltime[lmask], normed_lpower, lw=0.5, zorder=2, color='k')
    xmin, xmax, ymin, ymax = frame2.axis()
    frame2.set_ylim(ymin, ymax)
    frame2.hlines(faplevels, xmin, xmax,
                  colors='r', linestyles='dashed', zorder=1, alpha=0.5)
    frame2 = period_arrows(frame2, lx, ly, ymax, color='k', zorder=1)
    frame2 = period_arrows(frame2, mlx, mly, ymax, color='r', zorder=2)
    # ##########################################################################
    # This is a total HACK and has no real justification
    mltime = np.array(1.0 / mlfreq)
    mlmask = (mltime > limits[0]) & (mltime < limits[1])
    normed_mlpower = abs(mlpower[mlmask] - np.mean(mlpower[mlmask]))
    normed_mlpower = np.max(normed_lpower) * normed_mlpower / np.max(
        normed_mlpower)
    frame2.plot(mltime[mlmask], normed_mlpower, linestyle='--',
                lw=0.5, zorder=3, color='r')
    # ##########################################################################
    frame2.set_xlabel('time (d)')
    frame2.set_ylabel('$P_N$')
    frame2.set_title('Lomb-Scagle')
    frame2.set_xscale('log')
    frame2.set_xlim(*limits)

    # -------------------------------------------------------------------------
    # frame 3: Clean
    # -------------------------------------------------------------------------
    ctime = np.array(1.0 / cfreq)
    cmask = (ctime > limits[0]) & (ctime < limits[1])
    normed_cpower = cpower[cmask]
    frame3.plot(ctime[cmask], cpower[cmask], lw=0.5, zorder=2, color='k')
    xmin, xmax, ymin, ymax = frame3.axis()
    frame3.set_ylim(ymin, ymax)
    frame3 = period_arrows(frame3, cx, cy, ymax, color='k', zorder=1)
    # # ########################################################################
    # mctime = np.array(1.0/mcfreq)
    # mcmask = (mctime > limits[0]) & (mctime < limits[1])
    # # This is a total HACK and has no real justification
    # normed_mcpower = abs(mcpower[mcmask] - np.mean(mcpower[mcmask]))
    # normed_mcpower = np.max(normed_cpower)*normed_mcpower/np.max(normed_mcpower)
    # frame2.plot(mctime[mcmask], normed_mcpower, linestyle='--',
    #             lw=0.5, zorder=3, color='r')
    # # ########################################################################
    frame3.set_xlabel('time (d)')
    frame3.set_ylabel('$P_N$')
    frame3.set_title('Clean')
    frame3.set_xscale('log')
    frame3.set_xlim(*limits)

    # -------------------------------------------------------------------------
    # frame 4: table of periods
    ps, lps = [lx, cx], ['Lombscargle periods / days', 'Clean periods / days']
    nps, lnps = [mlx], ['Lombscargle noise periods / days']
    frame4 = plot_period_table(frame4, periods=ps, noise_periods=nps,
                               labelperiods=lps, labelnoiseperiods=lnps,
                               percentage=percentage)

    # -------------------------------------------------------------------------
    # frame 5+: Phase folded lightcurve
    # -------------------------------------------------------------------------
    for fit, frame in enumerate(frames):
        frame.errorbar(tfold[fit], data, yerr=edata, linestyle='None',
                        marker='o', ms=4, color='k')
        frame.plot(tfit[fit], dfit[fit], color='r')
        frame.set_xlabel('rotation phase')
        frame.set_ylabel('WASP V mag')
        frame.set_ylim(*frame.get_ylim()[::-1])
        frame.set_title('Folded on {0}'.format(glpeaks[fit]))

    # -------------------------------------------------------------------------
    # save show close
    plt.subplots_adjust(hspace=0.3, wspace=0.3, top=0.95, bottom=0.01,
                        right=0.95, left=0.1)
    if show:
        plt.show()
        plt.close()
    else:
        sname = 'Messina_plot_' + name.replace(' ', '_')
        plt.savefig(savepath + sname + '.png', bbox_inches='tight')
        plt.savefig(savepath + sname + '.pdf', bbox_inches='tight')
        plt.close()


def plot_period_table(frame, periods, noise_periods, labelperiods,
                      labelnoiseperiods, percentage=1.0):
    frame.axis('off')

    calcs = select_matching_periods(periods, noise_periods, labelperiods,
                                    labelnoiseperiods, percentage)
    cell_text, cell_colour, numperiod, numnoise, row_names = calcs
    cols = list(labelperiods) + list(labelnoiseperiods)
    colws = [0.2] * (numperiod + numnoise)

    the_table = frame.table(cellText=cell_text, cellColours=cell_colour,
                            rowLabels=row_names,
                            colLabels=cols, colWidths=colws,
                            loc='center')
    the_table.scale(1, 3)
    return frame


def select_matching_periods(periods, noise_periods, labelperiods=None,
                            labelnoiseperiods=None, percentage=1.0):
    threshold = percentage / 100.0
    numperiod, numnoise = len(periods), len(noise_periods)
    tdata = list(periods) + list(noise_periods)
    tdata = np.array(tdata).T
    goodmatches, goodmatches_index = [], []
    cell_text, row_names, cell_colour = [], [], []
    # Loop round each row
    for row in range(len(tdata)):
        row_names.append('Peak {0}'.format(row + 1))
        cell_row_text, cell_row_colour = [], []
        # loop round each column
        for col in range(len(tdata[row])):
            cellvalue = tdata[row][col]
            cell_row_text.append('{0:.8f}'.format(cellvalue))
            if col >= numperiod:
                cell_row_colour.append('w')
                continue
            # if col within threshold of noise row then shade grey
            low = (1.0 - threshold) * cellvalue
            high = (1.0 + threshold) * cellvalue
            cond1 = tdata[:, numperiod:] > low
            cond2 = tdata[:, numperiod:] < high
            cond3 = tdata[:, :numperiod] > low
            cond4 = tdata[:, :numperiod] < high
            if np.sum(cond1 & cond2) > 1:
                cell_row_colour.append('0.5')
            # else if period within 10% of other period shade yellow
            elif np.sum(cond3 & cond4) > 1:
                cell_row_colour.append('y')
                goodmatches.append(cellvalue)
                goodmatches_index.append(col)
            else:
                cell_row_colour.append('w')
        # append to
        cell_text.append(cell_row_text)
        cell_colour.append(cell_row_colour)

    if labelnoiseperiods is None or labelperiods is None:
        return goodmatches, goodmatches_index
    else:
        full = [cell_text, cell_colour, numperiod, numnoise, row_names]
        return full


def label_select_matching_periods(percentage, goodms, goodmis):
    goodms = np.array(goodms)
    goodmis = np.array(goodmis)
    # if goodms is empty we cannot find a peak in common
    if len(goodms) == 0:
        return [], []
    # use relative clustering algorithm to group periods (by percentage)
    groups, groupindexs = relative_cluster(goodms, percentage,
                                           return_indices=True)
    # loop round each group, choose a representative period and make label
    goodpeaks, goodpeaklabels = [], []
    for git, group in enumerate(groups):
        # pick one in group
        goodpeak = np.random.choice(group)
        goodpeaklabel = ''
        # label is combination of all in group
        for p, peak in enumerate(group):
            # get original index
            original_index = groupindexs[git][p]
            # flag as choosen peak
            if peak == goodpeak:
                choosen = r'$^{*}$'
            else:
                choosen = ''
            # find out whether peak is a lomb or a clean
            if goodmis[original_index] == 0:
                kind = 'Lombscargle'
            else:
                kind = 'Clean'
            # append group peak to good peak label
            if p != 0:
                goodpeaklabel += ', '
            goodpeaklabel += 'Good {0} p={1:.4f} days'.format(kind, peak)
            goodpeaklabel += choosen
        # finally append good peak and label to arrays
        goodpeaks.append(goodpeak), goodpeaklabels.append(goodpeaklabel)
    return goodpeaks, goodpeaklabels


def period_arrows(frame, xarr=None, yarr=None, height=0.1, **kwargs):
    if xarr is None or yarr is None:
        return frame

    colour = kwargs.get('color', 'b')
    zorder = kwargs.get('zorder', 1)

    opt = dict(color=colour, zorder=zorder,
               arrowstyle='simple,head_width=.25,head_length=.25',
               connectionstyle='arc3,rad=0')

    wspace = height * 1.1
    for i in range(len(xarr)):
        xy = [xarr[i], wspace]
        xytext = [xarr[i], 1.01 * wspace]
        frame.annotate('', xy=xy, xycoords='data', xytext=xytext,
                       textcoords='data', arrowprops=opt, size=20)

    xmin, xmax, ymin, ymax = frame.axis()
    frame.set_ylim(0, np.max([height * 1.1 * 1.1, ymax]))
    # frame.arrow(xarr[i], yarr[i]+wspace, dx=0, dy=height, **kwargs)
    return frame


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # load arguments from command line or defaults
    pp = load_params()
    # ----------------------------------------------------------------------
    # if plot exists then skip (if SKIP = True)
    tname = pp['SID']
    if pp['SKIP']:
        filename = 'Messina_plot_' + tname.replace(' ', '_') + '.png'
        if filename in os.listdir(pp['PLOTPATH']):
            sys.exit('Skipping due to file existing')
    # ----------------------------------------------------------------------
    # load data from file
    if '.fits' in pp['DPATH']:
        lightcurve = fits.getdata(pp['DPATH'], ext=1)
    elif pp['DPATH'] != str(None):
        print("\n Loading data...")
        path = pp['DPATH'] + '{0}_lightcurve.fits'.format(pp['SID'])
        lightcurve = fits.getdata(path)
    # if not load from database
    else:
        sql_kwargs = dict(host=HOSTNAME, db=DATABASE, table=TABLE,
                          user=USERNAME,
                          passwd=PASSWORD, conn_timeout=100000)
        lightcurve = get_lightcurve_data(conn=None, sid=pp['SID'],
                                         sortcol=pp['TIMECOL'], **sql_kwargs)
    # ----------------------------------------------------------------------
    # get columns
    time_arr = np.array(lightcurve[pp['TIMECOL']], dtype=float)
    time_arr -= pp['TIME_CONST']
    data_arr = np.array(lightcurve[pp['DATACOL']], dtype=float)
    edata_arr = np.array(lightcurve[pp['EDATACOL']], dtype=float)
    # ----------------------------------------------------------------------
    # if data is empty raise exception
    nanmask = np.isfinite(time_arr) & np.isfinite(data_arr)
    if np.sum(nanmask) == 0:
        raise IndexError('Time array / Data array have zero real values.')
    # ----------------------------------------------------------------------
    # clean data
    nkwargs = dict(bindata=pp['BINDATA'], binsize=pp['BINSIZE'],
                   sigmaclip=pp['SIGMACLIP'], sigma=pp['SIGMA'],
                   size=pp['SIZE'], errorclip=pp['ERRORCLIP'],
                   percentage=pp['PERCENTAGE'])
    time_arr, data_arr, edata_arr = neil_clean(time_arr, data_arr, edata_arr,
                                               **nkwargs)
    # -------------------------------------------------------------------------
    # Run clean periodogram
    ckwargs = dict(freqs=None, log=True, full=True, maxsize=10000,
                   fmax=pp['NYQUIST_FACTOR'], ppb=pp['SAMPLES_PER_PEAK'])
    freqs1, wfn, dft, cdft = clean_periodogram(time_arr, data_arr, **ckwargs)
    # cdft is the amplitudes power = DFT(x) * conj(DFT(x))
    cpower = np.array(cdft * np.conjugate(cdft))
    freqs1a = freqs1[0: len(cpower)]
    # -------------------------------------------------------------------------
    # Cannot do currently as frequencies different for each iteration of the
    # monte carlo --> how do we get around this? intrepret onto grid?
    # this is really slow too

    # # Run Monte Carlo clean periodogram
    # fargs = [time_arr, data_arr]
    # fkwargs = dict(freqs=None, log=True, full=True, maxsize=10000,
    #                fmax=NYQUIST_FACTOR, ppb=SAMPLES_PER_PEAK)
    # lmkwargs = dict(N=100, log=True, nyquist_factor=NYQUIST_FACTOR,
    #                 samples_per_peak=SAMPLES_PER_PEAK)
    # freqs1_mc, cdft_mc, _, _ = fap_montecarlo(clean_periodogram, fargs,
    #                                           fkwargs, **lmkwargs)
    # # cdft is the amplitudes power = DFT(x) * conj(DFT(x))
    # cpower_mc = np.array(cdft_mc*np.conjugate(cdft_mc))
    # freqs1a_mc = freqs1_mc[0: len(cpower_mc)]
    freqs1a_mc, cpower_mc = None, None
    # -------------------------------------------------------------------------
    # Run lombscargle periodogram
    lkwargs = dict(freqs=None, nyquist_factor=pp['NYQUIST_FACTOR'],
                   samples_per_peak=pp['SAMPLES_PER_PEAK'])
    freqs2, lpower = lombscargle_periodogram(time_arr, data_arr, edata_arr,
                                             **lkwargs)
    # -------------------------------------------------------------------------
    # Run Monte Carlo lombscargle periodogram
    fargs = [time_arr, data_arr, edata_arr]
    fkwargs = dict(freqs=None)
    lmkwargs = dict(N=100, log=True, nyquist_factor=pp['NYQUIST_FACTOR'],
                    samples_per_peak=pp['SAMPLES_PER_PEAK'])
    freqs2_mc, lpower_mc, _, _ = fap_montecarlo(lombscargle_periodogram, fargs,
                                                fkwargs, **lmkwargs)
    # -------------------------------------------------------------------------
    # find peaks of the lombscargle periodogram
    peakkwargs = dict(number=10, x_range=pp['PEAK_RANGE'])
    lxpeak, lypeak = find_y_peaks(1.0 / freqs2, lpower, **peakkwargs)
    # -------------------------------------------------------------------------
    # find peaks of the lombscargle monte carlo data
    lxpeak_mc, lypeak_mc = find_y_peaks(1.0 / freqs2_mc, lpower_mc,
                                        **peakkwargs)
    # -------------------------------------------------------------------------
    # find peaks of the clean periodogram
    cxpeak, cypeak = find_y_peaks(1.0 / freqs1a, cpower.real, **peakkwargs)
    # -------------------------------------------------------------------------
    # find peaks of the clean monte carlo data
    # cxpeak_mc, cypeak_mc = find_y_peaks(1.0/freqs1a_mc, cpower_mc.real,
    #                                     **peakkwargs)
    cxpeak_mc, cypeak_mc = None, None
    # -------------------------------------------------------------------------
    # Select periods (use messina and find periods from periodograms)
    if pp['MESSINA_PERIOD'] is not None:
        goodpeaks = [pp['MESSINA_PERIOD']]
        gpargs = pp['MESSINA_PERIOD']
        goodpeaklabels = ['Messina period = {0} days'.format(gpargs)]
    else:
        goodpeaks = []
        goodpeaklabels = []
    # select good and bad peaks from matching periods
    spercentage = pp['PEAK_THRESHOLD_PERCENTAGE']
    sresults = select_matching_periods([lxpeak, cxpeak], [lxpeak_mc],
                                       percentage=spercentage)
    sresults = [spercentage] + [sresults[0]] + [sresults[1]]
    goodpeaks1, goodpeaklabels1 = label_select_matching_periods(*sresults)
    # add periods to messina (if used)
    goodpeaks += goodpeaks1
    goodpeaklabels += goodpeaklabels1
    # -------------------------------------------------------------------------
    # phase folding
    timefold, timefit, datafit = phase_fold(time_arr, data_arr, goodpeaks)
    # -------------------------------------------------------------------------
    # plot messina plot
    plotargs = [time_arr, data_arr, edata_arr, tname, freqs2, lpower,
                freqs2_mc, lpower_mc, freqs1a, cpower, freqs1a_mc, cpower_mc,
                timefold, timefit, datafit, lxpeak, lypeak, lxpeak_mc,
                lypeak_mc, cxpeak, cypeak, cxpeak_mc, cypeak_mc, goodpeaklabels]
    pp['show'] = False
    messina_plot(*plotargs, **pp)


# =============================================================================
# End of code
# =============================================================================
