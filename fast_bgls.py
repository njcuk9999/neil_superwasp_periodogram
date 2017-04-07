# -*- coding: utf-8 -*-
# ================================================================================

# Modified version of:

# Copyright (c) 2014 João Faria, Annelies Mortier
# Distributed under the MIT License.
# (See accompanying file LICENSE or copy at http://opensource.org/licenses/MIT)
# ================================================================================

import numpy as np
import sys
try:
    import mpmath  # https://code.google.com/p/mpmath/
except ImportError as e1:
    try:
        from sympy import mpmath  # http://sympy.org/en/index.html
    except ImportError as e2:
        raise e2
    finally:
        raise e1

pi = np.pi


def bgls(t, y, dy, plow=0.5, phigh=100, ofac=1, samples_per_peak=None,
         freq=None):
    """
    Computes the Bayesian Generalised Lomb-Scargle Periodogram

    :param t: numpy array, time vector
    :param y: numpy array, data vector
    :param dy: numpy array, uncertainties associated with data vector
    :param plow: float, minimum period to find
    :param phigh: float, maximum period to find
    :param ofac: int, oversampling frequency
    :param samples_per_peak: int or None, if not None number of samples per peak
    :param freq: numpy array or None, if not None sets the frequency grid to use
    :return: 
    """
    # Calculate frequency grid (or use one given)
    if freq is not None:
        f = freq
    elif samples_per_peak is not None:
        baseline = t.max() - t.min()
        df = 1. / baseline / samples_per_peak
        Nf = int(np.ceil((1. / plow - 1. / phigh) / df))
        f = np.linspace(1. / phigh, 1. / plow, Nf)
    else:
        Nf = int(100 * ofac)
        f = np.linspace(1. / phigh, 1. / plow, Nf)
    # calculate constants and exponents
    constants, exponents = core1(t, y, dy, f)
    # convert constants and exponents to numpy arrays
    constants = np.array(constants)
    exponents = np.array(exponents)
    # calculate the log probability
    # log10(prob) =  log10(constants) + exponents * log10(exp(1))
    log10prob = np.log10(constants) + (exponents * np.log10(np.exp(1.)))
    # probability = 10^(lnprob)
    prob = [10 ** mpmath.mpf(x) for x in log10prob]
    # normalise the probability to the maximum probability ???
    prob = np.array(prob) / max(prob)  # normalize
    # very small probabilities need to be set to zero
    prob[prob < (sys.float_info.min * 10)] = 0
    prob = np.array([float(pp) for pp in prob])
    # return the period grid and the probabilities
    return 1. / f, prob


def bgls_fast(t, y, dy, plow=0.5, phigh=100, ofac=1, samples_per_peak=None,
         freq=None, maxsize=5000, log=False):
    """
    Computes the Bayesian Generalised Lomb-Scargle Periodogram

    :param t: numpy array, time vector
    :param y: numpy array, data vector
    :param dy: numpy array, uncertainties associated with data vector
    :param plow: float, minimum period to find
    :param phigh: float, maximum period to find
    :param ofac: int, oversampling frequency
    :param samples_per_peak: int or None, if not None number of samples per peak
    :param freq: numpy array or None, if not None sets the frequency grid to use
    :return:
    """
    # Calculate frequency grid (or use one given)
    if freq is not None:
        f = freq
    elif samples_per_peak is not None:
        baseline = t.max() - t.min()
        df = 1. / baseline / samples_per_peak
        Nf = int(np.ceil((1. / plow - 1. / phigh) / df))
        f = np.linspace(1. / phigh, 1. / plow, Nf)
    else:
        Nf = int(100 * ofac)
        f = np.linspace(1. / phigh, 1. / plow, Nf)
    # calculate constants and exponents using fast chunked method
    constants, exponents = core2chunks(t, y, dy, f, maxsize=maxsize, log=log)
    # convert constants and exponents to numpy arrays
    constants = np.array(constants)
    exponents = np.array(exponents)
    # calculate the log probability
    # log10(prob) =  log10(constants) + exponents * log10(exp(1))
    log10prob = np.log10(constants) + (exponents * np.log10(np.exp(1.)))
    # probability = 10^(lnprob)
    prob = [10 ** mpmath.mpf(x) for x in log10prob]
    # normalise the probability to the maximum probability ???
    prob = np.array(prob) / max(prob)  # normalize
    # very small probabilities need to be set to zero
    prob[prob < (sys.float_info.min * 10)] = 0
    prob = np.array([float(pp) for pp in prob])
    # return the period grid and the probabilities
    return 1. / f, prob


def core1(t, y, dy, f):
    """

    :param t: numpy array, time vector
    :param y: numpy array, data vector
    :param dy: numpy array, uncertainties associated with data vector
    :param f: numpy array, frequency grid
    :return:
    """
    # omega is 2pi*f
    omegas = 2. * pi * f
    # calcualte the weights
    w = np.array(1. / dy**2)
    # calculate the sum of the weights
    W = np.sum(w)
    # calculate the sum of the products of the weights and data values
    bigY = np.sum(w * y)  # Eq. (10)

    constants = []
    exponents = []

    for i, omega in enumerate(omegas):
        theta = 0.5 * np.arctan2(np.sum(w * np.sin(2. * omega * t)),
                                 np.sum(w * np.cos(2. * omega * t)))
        x = omega * t - theta
        cosx = np.cos(x)
        sinx = np.sin(x)
        wcosx = w * cosx
        wsinx = w * sinx
        C = np.sum(wcosx)
        S = np.sum(wsinx)
        YCh = np.sum(y * wcosx)
        YSh = np.sum(y * wsinx)
        CCh = np.sum(wcosx * cosx)
        SSh = np.sum(wsinx * sinx)
        if (CCh != 0 and SSh != 0):
            K = (C * C * SSh + S * S * CCh - W * CCh * SSh) / (2. * CCh * SSh)
            L = (bigY * CCh * SSh - C * YCh * SSh - S * YSh * CCh) / (CCh * SSh)
            M = (YCh * YCh * SSh + YSh * YSh * CCh) / (2. * CCh * SSh)
            constants.append(1. / np.sqrt(CCh * SSh * np.abs(K)))
        elif (CCh == 0):
            K = (S * S - W * SSh) / (2. * SSh)
            L = (bigY * SSh - S * YSh) / (SSh)
            M = (YSh * YSh) / (2. * SSh)
            constants.append(1. / np.sqrt(SSh * np.abs(K)))
        elif (SSh == 0):
            K = (C * C - W * CCh) / (2. * CCh)
            L = (bigY * CCh - C * YCh) / (CCh)
            M = (YCh * YCh) / (2. * CCh)
            constants.append(1. / np.sqrt(CCh * np.abs(K)))
        else:
            raise RuntimeError('CCh and SSh are both zero.')

        if K > 0:
            raise RuntimeError('K is positive. This should not happen.')

        exponents.append(M - L * L / (4. * K))
    return constants, exponents


def core2chunks(t, y, dy, f, maxsize=5000, log=False):
    chunks = int(np.ceil(len(f) / maxsize))
    constants, exponents = [], []
    for chunk in __tqdmlog__(range(chunks), log):
        # break frequency into bits
        fchunk = f[chunk * maxsize: (chunk + 1) * maxsize]
        # run core code on chunk
        ci, ei = core2(t, y, dy, fchunk)
        # add together
        constants = np.append(constants, ci)
        exponents = np.append(exponents, ei)
    return constants, exponents


def core2(ti, yi, dyi, fi):
    """

    :param t: numpy array, time vector
    :param y: numpy array, data vector
    :param dy: numpy array, uncertainties associated with data vector
    :param f: numpy array, frequency grid
    :return:
    """
    try:
        from numexpr import evaluate as ne
    except ModuleNotFoundError:
        return core1(ti, yi, dyi, fi)
    # -------------------------------------------------------------------------
    # convert t, y and dy to matrices
    ti = np.matrix(ti)
    yi = np.matrix(yi)
    dyi = np.matrix(dyi)
    # -------------------------------------------------------------------------
    # calcualte the weights
    w = 1. / np.multiply(dyi, dyi)
    # calculate the sum of the weights
    W = np.sum(w)
    # calculate the sum of the products of the weights and data values
    bigY = np.sum(np.multiply(w, yi))  # Eq. (10)
    # -------------------------------------------------------------------------
    # omega is 2pi*f
    omegas = np.matrix(2. * pi * fi).T
    # for i, omega in enumerate(omegas):
    # theta = 0.5 * np.arctan2(np.sum(w * np.sin(2. * omega * t)),
    #                          np.sum(w * np.cos(2. * omega * t)))
    sum1 = np.sum(ne('w*sin(2*omegas*ti)'), axis=1)
    sum2 = np.sum(ne('w*cos(2*omegas*ti)'), axis=1)
    theta = 0.5 * ne('arctan2(sum1, sum2)')
    # -------------------------------------------------------------------------
    # x = omega * t - theta
    x = np.subtract(ne('omegas*ti'), np.matrix(theta).T)
    # -------------------------------------------------------------------------
    # cosx = np.cos(x)
    cosx = ne('cos(x)')
    # sinx = np.sin(x)
    sinx = ne('sin(x)')
    # wcosx = w * cosx
    wcosx = ne('w * cosx')
    # wsinx = w * sinx
    wsinx = ne('w * sinx')
    # -------------------------------------------------------------------------
    # C = np.sum(wcosx)
    C = np.sum(wcosx, axis=1)
    # S = np.sum(wsinx)
    S = np.sum(wsinx, axis=1)
    # YCh = np.sum(y * wcosx)
    YCh = np.sum(ne('yi * wcosx'), axis=1)
    # YSh = np.sum(y * wsinx)
    YSh = np.sum(ne('yi * wsinx'), axis=1)
    # CCh = np.sum(wcosx * cosx)
    CCh = np.sum(ne('wcosx * cosx'), axis=1)
    # SSh = np.sum(wsinx * sinx)
    SSh = np.sum(ne('wsinx * sinx'), axis=1)
    # -------------------------------------------------------------------------
    # CChSSh = CCh * SSh   if CCh and SSh are not zero
    # CChSSh = SSh   if CCh = 0
    # CChSSh = CCh   if SSh = 0
    CChSSh = np.zeros_like(CCh)
    # CChSSh = CCh * SSh   if CCh and SSh are not zero
    mask1 = (CCh != 0) & (SSh != 0)
    CChSSh[mask1] = ne('CCh * SSh')[mask1]
    # CChSSh = SSh   if CCh = 0
    mask2 = (CCh == 0)
    CChSSh[mask2] = SSh[mask2]
    # CChSSh = CCh   if SSh = 0
    mask3 = (SSh == 0)
    CChSSh[mask3] = CCh[mask3]
    # -------------------------------------------------------------------------
    # K = (C * C * SSh + S * S * CCh - W * CCh * SSh) / (2. * CCh * SSh)
    K = ne('(C * C * SSh + S * S * CCh - W * CCh * SSh) / (2. * CChSSh)')
    # L = (bigY * CCh * SSh - C * YCh * SSh - S * YSh * CCh) / (CCh * SSh)
    L = ne('(bigY * CCh * SSh - C * YCh * SSh - S * YSh * CCh) / (CChSSh)')
    # M = (YCh * YCh * SSh + YSh * YSh * CCh) / (2. * CCh * SSh)
    M = ne('(YCh * YCh * SSh + YSh * YSh * CCh) / (2. * CChSSh)')
    # -------------------------------------------------------------------------
    # constants.append(1. / np.sqrt(CCh * SSh * np.abs(K)))
    constants = 1./ ne('sqrt(CChSSh * abs(K))')
    exponents = ne('M-((L*L)/(4.0 * K))')
    #
    # if K > 0:
    #     raise RuntimeError('K is positive. This should not happen.')
    # clean up
    del w, W, bigY, omegas, sum1, sum2, theta, x
    del cosx, sinx, wcosx, wsinx, C, S, YCh, YSh, CCh, SSh
    del CChSSh, mask1, mask2, mask3, K, L, M
    # return
    return np.array(constants), np.array(exponents)


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



if __name__ == '__main__':
    import time as time_module
    start1 = time_module.time()
    periods1, probs1 = bgls(t, y, dy, freq=lsfreq)
    end1 = time_module.time()
    print('Time taken = {0}s'.format(end1 - start1))

    start2 = time_module.time()
    periods2, probs2 = bgls_fast(t, y, dy, maxsize=5000, log=True, freq=lsfreq)
    end2 = time_module.time()
    print('Time taken = {0}s'.format(end2 - start2))
