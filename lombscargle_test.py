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
    from periodogram_functions import lombscargle_periodogram
except ModuleNotFoundError:
    raise Exception("Program requires 'periodogram_functions.py'")

# =============================================================================
# Define variables
# =============================================================================
# set file paths
WORKSPACE = '/Astro/Projects/RayPaul_Work/SuperWASP/'
DPATH = WORKSPACE + 'Data/from_exoplanetarchive/'
DPATH += '1SWASP J192338.19-460631.5.fits'
PLOTPATH = WORKSPACE + '/Plots/Messina_like_plots_from_exoarchive/'
# -----------------------------------------------------------------------------
# Column info
TIMECOL = 'HJD'
DATACOL = 'MAG2'
EDATACOL = 'MAG2_ERR'

# nquist to test
nf_arr = [1, 2, 5, 10, 20, 50]
#nf_arr = [5]
# samples per peak to test
ssp_arr = [1, 2, 5, 10, 20, 50]
# ssp_arr = [5]
# dfs to use
# df_arr = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
df_arr = [None]

# =============================================================================
# Define functions
# =============================================================================
def get_freq(df, N):
    if df is None:
        return None
    freq = []
    for i in range(N):
        freq.append(df * (N - i))
    return np.sort(freq)


def ls_test(time_arr, data_arr, edata_arr, freq, nf, ssp):

    lkwargs = dict(freqs=freq, nyquist_factor=nf, samples_per_peak=ssp)
    freqs, lpower = lombscargle_periodogram(time_arr, data_arr, edata_arr,
                                             **lkwargs)
    return freqs, lpower


def grid_plot(combinations, freq_arr, lpower_arr):
    rows = int(np.ceil(np.sqrt(len(combinations))))
    plt.close()
    fig, frames = plt.subplots(nrows=rows, ncols=rows)
    ymin_all, ymax_all = 0.0, 0.0
    for row in tqdm(range(rows)):
        for col in range(rows):
            frame = frames[col][row]
            it = row + rows*col
            if it >= len(combinations):
                # frame.axis('off')
                continue
            else:
                combination = combinations[it]
                titlestring = 'nf={0} ssp={1} df={2}'.format(*combination)
                frame.plot(1.0 /freq_arr[it], lpower_arr[it])
                frame.text(0.5, 0.5, titlestring, horizontalalignment='center',
                           verticalalignment = 'center',
                           transform = frame.transAxes)
                xmin, xmax, ymin, ymax = frame.axis()
                if ymin < ymin_all:
                    ymin_all = float(ymin)
                if ymax > ymax_all:
                    ymax_all = float(ymax)
                if row != 0:
                    frame.set_ylabel('')
                    frame.set_yticklabels([])
                else:
                    frame.set_ylabel('Power')
                if col != rows-1:
                    frame.set_xlabel('')
                    frame.set_xticklabels([])
                else:
                    frame.set_xlabel('Time / days')

    print('\n Matching yaxis...')
    for row in tqdm(range(rows)):
        for col in range(rows):
            frame = frames[col][row]
            frame.set_ylim(ymin_all, ymax_all)
            frame.set_xlim(10**-3, 10**3)
            frame.set_xscale('log')
            frame.grid(True)

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()
    plt.close()
    return ymin_all, ymax_all


def bindata(x, y, bins):
    binned_data = []
    for it in range(len(bins[:-1])):
        mask = (x > bins[it]) & (x <= bins[it+1])
        if np.sum(mask) == 0:
            binned_data.append(0)
        else:
            binned_data.append(np.nanmedian(y[mask]))
    return np.array(binned_data)


def density_plot(combinations, freq_arr, lpower_arr, ymin_all, ymax_all,
                 vary_arr, variablelabel, independentlabel, axis=0):
    rows = int(np.ceil(np.sqrt(len(combinations))))
    lbins = np.linspace(-3.0, 3.0, 100)
    bins = 10 ** lbins
    vary_arr = np.array(vary_arr)
    # bin up the lpower_arr data in the nyquest factor direction
    # samples per peak varies along data
    data1 = []
    for row in tqdm(range(rows)):
        data_arr = np.zeros((rows, len(bins) - 1))
        for col in (range(rows)):
            if axis == 0:
                it = row + rows * col
            if axis == 1:
                it = col + rows * row
            if it >= len(combinations):
                continue
            bin_data = bindata(1.0 / freq_arr[it], lpower_arr[it], bins)
            # want data to be width wide and bindata long
            data_arr[col] = bin_data
        data1.append(data_arr)
    # plot density plots with varying samples per peak
    plt.close()
    fig, frames = plt.subplots(ncols=len(data1), nrows=1)
    for dt, data in enumerate(data1):
        extent = [lbins.min(), lbins.max(), 0, len(vary_arr)]
        frames[dt].imshow(data, aspect='auto', extent=extent,
                          vmin=ymin_all, vmax=ymax_all, origin='lower')
        frames[dt].set_xlabel('Time / days')
        xticks = frames[dt].get_xticks()
        xticklabels = [r'10$^{' + '{0:.1f}'.format(i) + r'}$' for i in xticks]
        frames[dt].set_xticklabels(xticklabels)
        frames[dt].set_yticklabels(ssp_arr)
        frames[dt].set_ylabel(variablelabel)
        frames[dt].set_title('{0} = {1}'.format(independentlabel, nf_arr[dt]))
    plt.show()
    plt.close()


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # loading data
    print('\n Loading data...')
    lightcurve = fits.getdata(DPATH, ext=1)
    # ----------------------------------------------------------------------
    # get columns
    time_arr = np.array(lightcurve[TIMECOL], dtype=float)
    data_arr = np.array(lightcurve[DATACOL], dtype=float)
    edata_arr = np.array(lightcurve[EDATACOL], dtype=float)
    N = len(time_arr)
    # ----------------------------------------------------------------------
    # make combinations of nf, ssp and df
    print('\n Calculating lombscargle for all combinations')
    freq_arr, lpower_arr = [], []
    combinations = list(itertools.product(nf_arr, ssp_arr, df_arr))
    for combination in tqdm(combinations):
        nf, ssp, df = combination
        freqs = get_freq(df, N)
        freqs, lpower = ls_test(time_arr, data_arr, edata_arr, freqs, nf, ssp)
        freq_arr.append(freqs), lpower_arr.append(lpower)
    # ----------------------------------------------------------------------
    # make grid graph
    print('\n Making graph grid')
    ymin_all, ymax_all = grid_plot(combinations, freq_arr, lpower_arr)
    # ----------------------------------------------------------------------
    # make density plots axis 0 = ssp, axis 1 = nf
    vary0 = 'Samples per peak'
    vary1 = 'Nyquist Frequency'
    print('\n Making density plot')
    density_plot(combinations, freq_arr, lpower_arr, ymin_all, ymax_all,
                 ssp_arr, vary0, vary1, axis=0)
    density_plot(combinations, freq_arr, lpower_arr, ymin_all, ymax_all,
                 nf_arr, vary1, vary0, axis=1)

# =============================================================================
# End of code
# =============================================================================
