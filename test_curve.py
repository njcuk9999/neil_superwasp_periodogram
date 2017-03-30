#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 30/03/17 at 1:56 PM

@author: neil

Program description here

Version 0.0.0
"""
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
try:
    from clean_periodogram import clean_periodogram
    from clean_periodogram import calc_freq
except ModuleNotFoundError:
    raise Exception("clean_periodogram.py needed")
try:
    from periodogram_functions import lombscargle_periodogram
    from periodogram_functions import fap_montecarlo
    from periodogram_functions import find_y_peaks
except ModuleNotFoundError:
    raise Exception("Program requires 'periodogram_functions.py'")

# =============================================================================
# Define variables
# =============================================================================
WORKSPACE = "/Astro/Projects/RayPaul_Work/SuperWASP/"
# -----------------------------------------------------------------------------
SID = 'GJ1289'
# SID = 'GJ793'
# -----------------------------------------------------------------------------
TIMECOL = "time"
DATACOL = "flux"
EDATACOL = "eflux"
# -----------------------------------------------------------------------------
# NYQUIST = 50
# SAMPLES = 5
NPEAKS = 50
DPEAKS = 10
PEAK_RANGE = [0.1, 100]
BOXSIZE = 10
PERCENTAGE = 10.0
# -----------------------------------------------------------------------------
# for GJ1289
if SID == 'GJ1289':
    DPATH = WORKSPACE + "Data/Elodie/bl_gj1289.fits"
    FREQS = np.arange(1.0/2274.961, 1.0/0.1053, 0.00044)
elif SID == 'GJ793':
    FREQS = np.arange(1.0/2364.844, 1.0/0.0985, 0.00042)
    DPATH = WORKSPACE + "Data/Elodie/bl_gj793.fits"

# =============================================================================
# Define functions
# =============================================================================
def plot(lsdata, mcdata, cdata, periodcols, periodcoll, nperiodcols,
         nperiodcoll, percentage, name, dpeak):

    plt.close()
    fig, frames = plt.subplots(ncols=2, nrows=2)
    # plot lombscargle periodogram
    frames[0][0].plot(1.0/lsdata[0], lsdata[1], color='b')
    frames[0][0].set_xlabel('Period / days')
    frames[0][0].set_ylabel('Power')
    frames[0][0].set_title('Lomb Scargle')
    frames[0][0].set_xscale('log')

    # plot CLEAN periodogram
    frames[0][1].plot(1.0/cdata[0], cdata[1], color='b')
    frames[0][1].set_xlabel('Period / days')
    frames[0][1].set_ylabel('Power')
    frames[0][1].set_title('CLEAN periodogram')
    frames[0][1].set_xscale('log')

    # plot MCMC noise periodogram
    frames[1][0].plot(1.0/mcdata[0], mcdata[1], color='r')
    frames[1][0].set_xlabel('Period / days')
    frames[1][0].set_ylabel('Power')
    frames[1][0].set_title('Noise periodogram')
    frames[1][0].set_xscale('log')

    # plot table
    frames[1][1] = plot_period_table(frames[1][1], periodcols, nperiodcols,
                                     periodcoll, nperiodcoll,
                                     percentage=percentage)

    plt.suptitle('Analysis for {0}'.format(name))
    plt.subplots_adjust(hspace=0.35, wspace=0.1)
    plt.show()
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
    the_table.scale(1.5, 1)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    return frame


def select_matching_periods(periods, noise_periods, labelperiods=None,
                            labelnoiseperiods=None, percentage=1.0,
                            showpeaks=10):
    threshold = percentage / 100.0
    numperiod, numnoise = len(periods), len(noise_periods)
    tdata = list(periods) + list(noise_periods)
    tdata = np.array(tdata).T
    goodmatches, goodmatches_index = [], []
    cell_text, row_names, cell_colour = [], [], []
    if len(tdata) < showpeaks:
        showpeaks = len(tdata)
    # Loop round each row
    for row in range(showpeaks):
        row_names.append('Peak {0}'.format(row + 1))
        cell_row_text, cell_row_colour = [], []
        # loop round each column
        for col in range(len(tdata[row])):
            cellvalue = tdata[row][col]
            cell_row_text.append('{0:.8f}'.format(cellvalue))
            if col >= numperiod:
                cell_row_colour.append('0.5')
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


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # Load data
    print('\n Loading data...')
    lightcurve = fits.getdata(DPATH, ext=1)
    name = DPATH.split('/')[-1].split('.fits')[0]
    # ----------------------------------------------------------------------
    # get columns
    time_arr = np.array(lightcurve[TIMECOL], dtype=float)
    data_arr = np.array(lightcurve[DATACOL], dtype=float)
    edata_arr = np.array(lightcurve[EDATACOL], dtype=float)
    # ----------------------------------------------------------------------
    # if data is empty raise exception
    nanmask = np.isfinite(time_arr) & np.isfinite(data_arr)
    if np.sum(nanmask) == 0:
        raise IndexError('Time array / Data array have zero real values.')
    time_arr -= np.nanmin(time_arr)

    # -------------------------------------------------------------------------
    # Calculate frequencies
    if FREQS is None:
        freqs = calc_freq(time_arr, df=None, fmax=None, ppb=None, dtmin=None)
    else:
        freqs = FREQS
    # -------------------------------------------------------------------------
    # Run clean periodogram
    ckwargs = dict(freqs=freqs, log=True, full=True, maxsize=10000)
    freqs1, wfn, dft, cdft = clean_periodogram(time_arr, data_arr, **ckwargs)
    # cdft is the amplitudes power = DFT(x) * conj(DFT(x))
    cpower = np.array(cdft * np.conjugate(cdft))
    freqs1a = freqs1[0: len(cpower)]
    cdata = [freqs1a, cpower.real]
    # -------------------------------------------------------------------------
    # Run lombscargle periodogram
    lkwargs = dict(freqs=freqs, norm=False)
    lsdata = lombscargle_periodogram(time_arr, data_arr, edata_arr,
                                     **lkwargs)
    # -------------------------------------------------------------------------
    # Run Monte Carlo lombscargle periodogram
    fargs = [time_arr, data_arr, edata_arr]
    fkwargs = dict(freqs=freqs, norm=False)
    lmkwargs = dict(N=1000, log=True)
    mcdata = fap_montecarlo(lombscargle_periodogram, fargs,
                                                fkwargs, **lmkwargs)
    # -------------------------------------------------------------------------
    # find peaks of the lombscargle periodogram
    peakkwargs = dict(number=NPEAKS, x_range=PEAK_RANGE, boxsize=BOXSIZE)
    lxpeak, lypeak = find_y_peaks(1.0 / lsdata[0], lsdata[1], **peakkwargs)
    # -------------------------------------------------------------------------
    # find peaks of the lombscargle monte carlo data
    lxpeak_mc, lypeak_mc = find_y_peaks(1.0 / mcdata[0], mcdata[1],
                                        **peakkwargs)
    # -------------------------------------------------------------------------
    # find peaks of the clean periodogram
    cxpeak, cypeak = find_y_peaks(1.0 / cdata[0], cdata[1], **peakkwargs)
    # -------------------------------------------------------------------------
    pcols = [lxpeak, cxpeak]
    pcoll = ['Lomb Scargle', 'CLEAN periodogram']
    npcols = [lxpeak_mc]
    npcoll = ['MCMC noise periodogram']

    plot(lsdata, mcdata, cdata, pcols, pcoll, npcols, npcoll, PERCENTAGE, name,
         DPEAKS)


# =============================================================================
# End of code
# =============================================================================
