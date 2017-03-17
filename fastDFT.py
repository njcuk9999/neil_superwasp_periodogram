#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/03/17 at 12:41 PM

@author: neil

Program description here

Version 0.0.0
"""

import numpy as np
from astropy.io import fits
from numexpr import evaluate as ne

# =============================================================================
# Define variables
# =============================================================================
WORKSPACE = '/Astro/Projects/RayPaul_Work/SuperWASP/'

# Test data 1
TESTPATH = WORKSPACE + '/Programs/CLEAN_periodogram_IDL/test.fits'

# -----------------------------------------------------------------------------
MAX_SIZE = 10000


# =============================================================================
# Define functions
# =============================================================================
def dft_o(freq, tvec, dvec, kind='half'):
    """
    Calculate the Discrete Fourier transform (slow scales with N^2)
    The DFT is normalised to have the mean value of the data at zero frequency

    :param freq: numpy array, frequency grid calculated from the time vector

    :param tvec: numpy array or list, input time(independent) vector, normalised
                 by the mean of the time vector

    :param dvec: numpy array or list, input dependent vector, normalised by the
                 mean of the data vector

    :param log: boolean, if True prints progress to standard output
                         if False silent

    :param kind: string, if 'half' uses only the largest half of the frequencies
                 if 'full' will return all frequencies

    :return wfn:  numpy array of complex numbers, spectral window function

    :return dft:  numpy array of complex numbers, "dirty" discrete Fourier
                  transform

    """
    # -------------------------------------------------------------------------
    # Code starts here
    # -------------------------------------------------------------------------
    wfn = np.zeros(len(freq), dtype=complex)
    if kind == 'half':
        dft = np.zeros(int(len(freq)/2), dtype=complex)
    else:
        dft = np.zeros(len(freq), dtype=complex)
    for i in range(len(freq)):
        phase = -2*np.pi*freq[i]*tvec
        phvec = np.array(np.cos(phase) + 1j * np.sin(phase))
        if kind == 'half':
            if i < int(len(freq)/2):
                wfn[i] = np.sum(phvec)/len(tvec)
                dft[i] = np.sum(dvec*phvec)/len(tvec)
            # complete the spectral window function
            else:
                wfn[i] = np.sum(phvec)/len(tvec)
        else:
            wfn[i] = np.sum(phvec) / len(tvec)
            dft[i] = np.sum(dvec * phvec) / len(tvec)
    return wfn, dft


def dft_ne(freq, tvec, dvec, log=False, kind='half'):
    """
    Calculate the Discrete Fourier transform (slow scales with N^2)
    The DFT is normalised to have the mean value of the data at zero frequency

    :param freq: numpy array, frequency grid calculated from the time vector

    :param tvec: numpy array or list, input time(independent) vector, normalised
                 by the mean of the time vector

    :param dvec: numpy array or list, input dependent vector, normalised by the
                 mean of the data vector

    :param log: boolean, if True prints progress to standard output
                         if False silent

    :param kind: string, if 'half' uses only the largest half of the frequencies
                 if 'full' will return all frequencies

    :return wfn:  numpy array of complex numbers, spectral window function

    :return dft:  numpy array of complex numbers, "dirty" discrete Fourier
                  transform

    """
    # -------------------------------------------------------------------------
    # Code starts here
    # -------------------------------------------------------------------------
    wfn = np.zeros(len(freq), dtype=complex)
    if kind == 'half':
        dft = np.zeros(int(len(freq)/2), dtype=complex)
    else:
        dft = np.zeros(len(freq), dtype=complex)
    # work out all ocnstants before loop
    Ntvec, two_pi_t = len(tvec), -2*np.pi*tvec
    Nfreq = int(len(freq)/2)
    # loop around freq
    for i in __tqdmlog__(range(len(freq)), log):
        freqi = freq[i]
        # phase = -2*np.pi*freq[i]*tvec
        # phvec = np.array(np.cos(phase) + 1j * np.sin(phase))
        phvec = ne('cos(freqi*two_pi_t) + 1j*sin(freqi*two_pi_t)')

        #wfn[i] =  np.sum(phvec) / len(tvec)
        wfn[i] = ne('sum(phvec/Ntvec)')
        if kind == 'half':
            if i < Nfreq:
                # dft[i] = np.sum(dvec*phvec)/len(tvec)
                dft[i] = ne('sum(dvec*phvec/Ntvec)')
        else:
            dft[i] = ne('sum(dev*phvec/Ntvec)')
    return wfn, dft


def dft_ne2(freq, tvec, dvec, kind='half'):
    """
    Calculate the Discrete Fourier transform (slow scales with N^2)
    The DFT is normalised to have the mean value of the data at zero frequency

    :param freq: numpy array, frequency grid calculated from the time vector

    :param tvec: numpy array or list, input time(independent) vector, normalised
                 by the mean of the time vector

    :param dvec: numpy array or list, input dependent vector, normalised by the
                 mean of the data vector

    :param kind: string, if 'half' uses only the largest half of the frequencies
                 if 'full' will return all frequencies

    :return wfn:  numpy array of complex numbers, spectral window function

    :return dft:  numpy array of complex numbers, "dirty" discrete Fourier
                  transform

    """
    # -------------------------------------------------------------------------
    # Code starts here
    # -------------------------------------------------------------------------
    wfn = np.zeros(len(freq), dtype=complex)
    if kind == 'half':
        dft = np.zeros(int(len(freq)/2), dtype=complex)
    else:
        dft = np.zeros(len(freq), dtype=complex)
    # work out all ocnstants before loop
    Ntvec, two_pi_t = len(tvec), -2*np.pi*tvec
    Nfreq = int(len(freq)/2)
    # loop around freq
    for i in range(len(freq)):
        freqi = freq[i]
        # phase = -2*np.pi*freq[i]*tvec
        phase = freqi*two_pi_t
        # phvec = np.array(np.cos(phase) + 1j * np.sin(phase))
        phvec = ne('cos(phase) + 1j*sin(phase)')
        #wfn[i] =  np.sum(phvec) / len(tvec)
        wfn[i] = ne('sum(phvec/Ntvec)')
        if kind == 'half':
            if i < Nfreq:
                # dft[i] = np.sum(dvec*phvec)/len(tvec)
                dft[i] = ne('sum(dvec*phvec/Ntvec)')
        else:
            dft[i] = ne('sum(dev*phvec/Ntvec)')

    return wfn, dft


def dft_nfor(freq, tvec, dvec, kind='half'):
    """
    Calculate the Discrete Fourier transform (slow scales with N^2)
    The DFT is normalised to have the mean value of the data at zero frequency

    :param freq: numpy array, frequency grid calculated from the time vector

    :param tvec: numpy array or list, input time(independent) vector, normalised
                 by the mean of the time vector

    :param dvec: numpy array or list, input dependent vector, normalised by the
                 mean of the data vector

    :param kind: string, if 'half' uses only the largest half of the frequencies
                 if 'full' will return all frequencies

    :return wfn:  numpy array of complex numbers, spectral window function

    :return dft:  numpy array of complex numbers, "dirty" discrete Fourier
                  transform

    """
    # -------------------------------------------------------------------------
    # Code starts here
    # -------------------------------------------------------------------------
    # wfn = np.zeros(len(freq), dtype=complex)
    # dft = np.zeros(int(len(freq)/2), dtype=complex)
    # make vectors in to matrices
    fmatrix = np.matrix(freq) # shape (N x 1)
    tmatrix = np.matrix(tvec) # shape (M x 1)
    # need dvec to be tiled len(freq) times (to enable multiplication)
    d_arr = np.tile(dvec, len(freq)).reshape(len(freq), len(dvec))
    dmatrix = np.matrix(d_arr)  # shape (N x M)
    # work out the phase
    phase = -2*np.pi*fmatrix.T*tmatrix  # shape (N x M)
    # work out the phase vector
    phvec = np.cos(phase) + 1j*np.sin(phase)   # shape (N x M)
    # for freq/2 rows
    wfn = np.sum(phvec, axis=1)/len(tvec)   # shape (N x 1)
    # only for the first freq/2 indices
    if kind == 'half':
        Nfreq = int(len(freq)/2)
        dft = np.sum(np.array(dmatrix)[: Nfreq] * np.array(phvec)[: Nfreq],
                     axis=1)/len(tvec)    # shape (N/2 x 1)
    else:
        dft = np.sum(np.array(dmatrix) * np.array(phvec), axis=1)/len(tvec)
    return wfn, dft



def dft_nfor_ne(freq, tvec, dvec, kind='half'):
    """
    Calculate the Discrete Fourier transform (slow scales with N^2)
    The DFT is normalised to have the mean value of the data at zero frequency

    :param freq: numpy array, frequency grid calculated from the time vector

    :param tvec: numpy array or list, input time(independent) vector, normalised
                 by the mean of the time vector

    :param dvec: numpy array or list, input dependent vector, normalised by the
                 mean of the data vector

    :param kind: string, if 'half' uses only the largest half of the frequencies
                 if 'full' will return all frequencies

    :return wfn:  numpy array of complex numbers, spectral window function

    :return dft:  numpy array of complex numbers, "dirty" discrete Fourier
                  transform

    """
    # -------------------------------------------------------------------------
    # Code starts here
    # -------------------------------------------------------------------------
    # wfn = np.zeros(len(freq), dtype=complex)
    # dft = np.zeros(int(len(freq)/2), dtype=complex)
    # make vectors in to matrices
    fmatrix = np.matrix(freq) # shape (N x 1)
    tmatrix = np.matrix(tvec) # shape (M x 1)
    # need dvec to be tiled len(freq) times (to enable multiplication)
    d_arr = np.tile(dvec, len(freq)).reshape(len(freq), len(dvec))
    dmatrix = np.matrix(d_arr)  # shape (N x M)
    # work out the phase
    ftmatrix = fmatrix.T
    twopi = 2*np.pi
    # phase = -2*np.pi*fmatrix.T*tmatrix  # shape (N x M)
    phase = ne('-twopi*ftmatrix*tmatrix')
    # work out the phase vector
    # phvec = np.cos(phase) + 1j*np.sin(phase)   # shape (N x M)
    phvec = ne('cos(phase) + 1j*sin(phase)')
    # for freq/2 rows
    wfn = np.sum(phvec, axis=1)/len(tvec)   # shape (N x 1)
    # only for the first freq/2 indices
    if kind == 'half':
        Nfreq = int(len(freq)/2)
        darray = np.array(dmatrix[: Nfreq])
        phvecarray = np.array(phvec)[: Nfreq]
    else:
        darray = np.array(dmatrix)
        phvecarray = np.array(phvec)
    # dft = np.sum(np.array(dmatrix)[: Nfreq] * np.array(phvec)[: Nfreq],
    #             axis=1)/len(tvec)    # shape (N/2 x 1)
    multiply = ne('darray * phvecarray')
    dft = np.sum(multiply, axis=1)/len(tvec)
    return wfn, dft



def dft_l(freq, tvec, dvec, log=False, maxsize=None, kind='half'):
    """
    Calculate the Discrete Fourier transform (slow scales with N^2)
    The DFT is normalised to have the mean value of the data at zero frequency

    :param freq: numpy array, frequency grid calculated from the time vector

    :param tvec: numpy array or list, input time(independent) vector, normalised
                 by the mean of the time vector

    :param dvec: numpy array or list, input dependent vector, normalised by the
                 mean of the data vector

    :param log: boolean, if True prints progress to standard output
                         if False silent

    :param maxsize: int, maximum number of frequency rows to processes,
                  default is 10,000 but large tvec/dvec array will use
                  a large amount of RAM (64*len(tvec)*maxsize bits of data)
                  If the program is using too much RAM, reduce "maxsize" or
                  bin up tvec/dvec

    :param kind: string, if 'half' uses only the largest half of the frequencies
                 if 'full' will return all frequencies

    :return wfn:  numpy array of complex numbers, spectral window function

    :return dft:  numpy array of complex numbers, "dirty" discrete Fourier
                  transform

    """
    if maxsize is None:
        maxsize = MAX_SIZE
    if len(freq) < maxsize:
        wfn, dft = dft_nfor(freq, tvec, dvec, kind)
    # Need to cut up frequencies into managable chunks (with a for loop)
    else:
        chunks = int(np.ceil(len(freq)/maxsize))
        wfn, dft = [], []
        for chunk in __tqdmlog__(range(chunks), log):
            # break frequency into bits
            freqi = freq[chunk*maxsize: (chunk+1)*maxsize]
            # get wfni and dfti for this chunk
            wfni, dfti = dft_nfor(freqi, tvec, dvec, kind)
            # append to list
            wfn = np.append(wfn, np.array(wfni))
            dft = np.append(dft, np.array(dfti))
            # clean up
            del freqi, wfni, dfti
        # convert to numpy array
        wfn, dft = np.array(wfn), np.array(dft)
    return wfn, dft


def dft_l_ne(freq, tvec, dvec, log=False, maxsize=None, kind='half'):
    """
    Calculate the Discrete Fourier transform (slow scales with N^2)
    The DFT is normalised to have the mean value of the data at zero frequency

    :param freq: numpy array, frequency grid calculated from the time vector

    :param tvec: numpy array or list, input time(independent) vector, normalised
                 by the mean of the time vector

    :param dvec: numpy array or list, input dependent vector, normalised by the
                 mean of the data vector

    :param log: boolean, if True prints progress to standard output
                         if False silent

    :param maxsize: int, maximum number of frequency rows to processes,
                  default is 10,000 but large tvec/dvec array will use
                  a large amount of RAM (64*len(tvec)*maxsize bits of data)
                  If the program is using too much RAM, reduce "maxsize" or
                  bin up tvec/dvec

    :return wfn:  numpy array of complex numbers, spectral window function

    :return dft:  numpy array of complex numbers, "dirty" discrete Fourier
                  transform

    """
    if maxsize is None:
        maxsize = MAX_SIZE
    if len(freq) < maxsize:
        wfn, dft = dft_nfor_ne(freq, tvec, dvec)
    # Need to cut up frequencies into managable chunks (with a for loop)
    else:
        chunks = int(np.ceil(len(freq)/maxsize))
        wfn, dft = [], []
        for chunk in __tqdmlog__(range(chunks), log):
            # break frequency into bits
            freqi = freq[chunk*maxsize: (chunk+1)*maxsize]
            # get wfni and dfti for this chunk
            wfni, dfti = dft_nfor_ne(freqi, tvec, dvec, kind=kind)
            # append to list
            wfn = np.append(wfn, np.array(wfni))
            dft = np.append(dft, np.array(dfti))
            # clean up
            del freqi, wfni, dfti
        # convert to numpy array
        wfn, dft = np.array(wfn), np.array(dft)
    return wfn, dft


def __tqdmlog__(x_input, log):
    """
    Private function for dealing with logging

    :param x_input:  any iterable object

    :param log: bool, if True and module tqdm exists use logging

    :return:
    """
    # deal with importing tqdm
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        tqdm = (lambda x: x)
    # deal with logging
    if log:
        rr = tqdm(x_input)
    else:
        rr = x_input
    return rr


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    import time as ttime
    # Load data
    print('\n Loading data...')
    fitsrec = fits.getdata(TESTPATH, ext=1)
    time_arr = np.array(fitsrec['time'], dtype=np.float64)
    data_arr = np.array(fitsrec['flux'], dtype=np.float64)
    # define freqs
    times1, times2, times3, times4 = [], [], [], []
    lengths = []
    values = [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4]
    # values = [1e3]
    for val in values:
        interval = 2.5/val
        freqs = np.arange(0, 200.5 + interval, interval)
        lengths.append(len(freqs))
        # ----------------------------------------------------------------------
        print('\n\t Slow DFT...')
        start1 = ttime.time()
        wfn1, dft1 = dft_o(freqs, time_arr, data_arr)
        end1 = ttime.time()
        print('len(t)={0} len(f)={1}'.format(len(time_arr), len(freqs)))
        print('Time taken = {0}'.format(end1 - start1))
        times1.append(end1 - start1)
        # # ----------------------------------------------------------------------
        print('\n\t DFT with numexpr...')
        start2 = ttime.time()
        wfn2, dft2 = dft_l(freqs, time_arr, data_arr)
        end2 = ttime.time()
        print('len(t)={0} len(f)={1}'.format(len(time_arr), len(freqs)))
        print('Time taken = {0}'.format(end2 - start2))
        times2.append(end2 - start2)
        # # ----------------------------------------------------------------------
        print('\n\t DFT without for loop...')
        start3 = ttime.time()
        wfn3, dft3 = dft_l(freqs, time_arr, data_arr)
        end3 = ttime.time()
        print('len(t)={0} len(f)={1}'.format(len(time_arr), len(freqs)))
        print('Time taken = {0}'.format(end3 - start3))
        times3.append(end3 - start3)
        # ----------------------------------------------------------------------
        print('\n\t DFT without for loop + numexpr...')
        start4 = ttime.time()
        wfn4, dft4 = dft_l_ne(freqs, time_arr, data_arr)
        end4 = ttime.time()
        print('len(t)={0} len(f)={1}'.format(len(time_arr), len(freqs)))
        print('Time taken = {0}'.format(end4 - start4))
        times4.append(end4 - start4)
        # ----------------------------------------------------------------------

    import matplotlib.pyplot as plt
    plt.plot(lengths, times1, color='k', label='for loop')
    plt.plot(lengths, times2, color='b', label='for loop with numexpr')
    plt.plot(lengths, times3, color='r', label='no for loop')
    plt.plot(lengths, times4, color='purple', label='no for loop with numexpr')
    xmin, xmax, ymin, ymax = plt.gca().axis()
    plt.xscale('log')
    plt.legend(loc=0)
    plt.xlabel('Values of iteration')
    plt.ylabel('Time taken / s')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()
    plt.close()


# =============================================================================
# End of code
# =============================================================================

# lprun commands

# %load_ext line_profiler

# %lprun -s -f dft_nfor -T lp_results.txt dft_nfor(freq, tvec, dvec)

# %lprun -s -f DFT -T lp_results1.txt DFT(freq, tvec, dvec)