#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12/04/17 at 11:17 AM

@author: neil

Program description here

Version 0.0.0
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
import string
import itertools


# =============================================================================
# Define variables
# =============================================================================
WORKSPACE = '/Astro/Projects/RayPaul_Work/SuperWASP/'
DPATH = WORKSPACE + 'Data/ls_analysis_run/'
DFILE1 = DPATH + '/light_curve_analysis_periods_regions.fits'
DFILE2 = DPATH + '/light_curve_analysis_periods_regions_groups_5.fits'

# -----------------------------------------------------------------------------
THRESHOLD = 5.0
TOP_N = 3

# =============================================================================
# Define functions
# =============================================================================
def get_numeric_id_from_name(names):
    # strip Full and R_#
    raw_names = []
    for name in names:
        raw_name = name.split('_Full')[0].split('_R')[0]
        raw_names.append(raw_name)
    unames = np.unique(raw_names)
    uids = []
    for raw_name in raw_names:
        uid = np.where(np.in1d(unames, raw_name))[0][0]
        uids.append(uid + 100000)
    return uids


def get_subtype(names):

    full = np.zeros_like(names, dtype=bool)
    sub = np.zeros_like(names, dtype=bool)
    for n_it, name in enumerate(names):
        if "Full" in name:
            full[n_it] = True
        else:
            sub[n_it] = True
    return full, sub

def group_objects(names, uids):

    group_names, group_ids = dict(), dict()
    for n_it, name in enumerate(names):
        key = uids[n_it]
        if key not in group_names:
            group_names[key] = [name]
            group_ids[key] = [n_it]
        else:
            group_names[key].append(name)
            group_ids[key].append(n_it)
    # get group masks
    group_masks = dict()
    for group in group_names:
        gnames = group_names[group]
        group_mask = np.in1d(names, gnames)
        group_masks[group] = group_mask
    # get group lengths
    group_lengths = dict()
    for group in group_names:
        group_lengths[group] = len(group_names[group])

    return group_names, group_ids, group_masks, group_lengths


def score_peaks(data, gmask, numpeaks, numregions, threshold):

        top_n = TOP_N

        # ----------------------------------------------------------------------
        # peaks
        #  Full Peak 0     R1 Peak 0     R2 Peak 0   ...   R_N Peak 0
        #  Full Peak 1     ...           ...         ...       ...
        #  Full peak 2     ...           ...         ...       ...
        #  ...             ...           ...         ...       ...
        #  full peak M     ...           ...         ...   R_N Peak M

        # get the full set of peaks for this group (peak 0 -> M)
        # Full + R1 -> R_N
        peaks, powers, npeaks, npowers = [], [], [], []
        for m in range(numpeaks):
            peakm = np.array(data['Peak{0}'.format(m+ 1)])[gmask]
            peaks.append(peakm)
            powerm = np.array(data['Power{0}'.format(m+1)])[gmask]
            powers.append(powerm)
            npeakm = np.array(data['nPeak{0}'.format(m + 1)])[gmask]
            npeaks.append(npeakm)
            npowerm = np.array(data['nPower{0}'.format(m+1)])[gmask]
            npowers.append(npowerm)
        peaks, npeaks = np.array(peaks), np.array(npeaks)
        powers, npowers = np.array(powers), np.array(npowers)

        # all peaks left will have no be near any peak in "npeaks" and will
        # appear at least twice in "peaks"
        flatpeaks = peaks.flatten()
        flatx = np.indices(peaks.shape)[0].flatten()
        flaty = np.indices(peaks.shape)[1].flatten()
        peak_cg, cgi = cluster(flatpeaks, percentagemaxgap=threshold,
                          return_indices=True)
        # loop around peak_cg (and cgi) remove clusters with one object
        cpeak, cindex = [], []
        for i in range(len(cgi)):
            if len(peak_cg[i]) > 1:
                cpeak.append(peak_cg[i])
                cindex.append(cgi[i])

        # get the powers associated with each cluster group (from original
        # positions power array
        cpower = []
        # loop around each group
        for i in range(len(cindex)):
            cpower.append([])
            # loop around the members in this group
            for j in range(len(cindex[i])):
                pos = cindex[i][j]
                cpower[i].append(powers[flatx[pos], flaty[pos]])

        # get the median period and the sum of the powers for each cluster
        # group
        cmedian_period, cstd_period, csum_power = [], [], []
        for i in range(len(cindex)):
            cmedian_period.append(np.median(cpeak[i]))
            cstd_period.append(np.std(cpeak[i]))
            csum_power.append(np.sum(cpower[i]))

        # sort these by maximum power
        sort = np.argsort(csum_power)[::-1]
        spower = np.array(csum_power)[sort]
        speriod = np.array(cmedian_period)[sort]
        s_e_period = np.array(cstd_period)[sort]

        # save the top_n periods (if less add nans)
        if len(speriod) > top_n:
            save_period = speriod[:top_n]
            save_power = spower[:top_n]
            save_e_period = s_e_period[:top_n]
        else:
            nans = np.repeat([np.nan], top_n - len(speriod))
            save_period = np.append(speriod, nans)
            save_power = np.append(spower, nans)
            save_e_period = np.append(s_e_period, nans)
        # return top_n periods, powers and spread
        return save_period, save_power, save_e_period


def cluster(data, maxgap=None, percentagemaxgap=None,
            return_indices=False):
    """

    Arrange data into groups where successive elements
    differ by no more than *maxgap* or *percentagemaxgap*

    :param data: numpy array, vector to sort into groups

    :param maxgap: float or None, the maximum gap between groups of points

    :param percentagemaxgap: float or None, if not None overrides maxgap, the
                             relative maximum gap between groups defined as a
                             percetnage away from the data element

    :param return_indices: boolean, if True returns indices of the clustered
                           points

    modified from
       http://stackoverflow.com/questions/14783947/
               grouping-clustering-numbers-in-python

    example:

        cluster([1, 6, 9, 100, 102, 105, 109, 134, 139], maxgap=10)

        [[1, 6, 9], [100, 102, 105, 109], [134, 139]]

        cluster([1, 6, 9, 99, 100, 102, 105, 134, 139, 141], maxgap=10)

        [[1, 6, 9], [99, 100, 102, 105], [134, 139, 141]]

    """
    if maxgap is None and percentagemaxgap is None:
        raise ValueError(
            "Must define either maxgap or percentagemaxgap")

    data = np.array(data)
    argsort = np.argsort(data)
    data = data[argsort]

    groups = [[data[0]]]
    group_index = [[argsort[0]]]
    for i, x in enumerate(data[1:]):
        if percentagemaxgap is not None:
            gap = x * percentagemaxgap / 100.0
        else:
            gap = maxgap

        if abs(x - groups[-1][-1]) <= gap:
            groups[-1].append(x)
            group_index[-1].append(argsort[i + 1])
        else:
            groups.append([x])
            group_index.append([argsort[i + 1]])
    if return_indices:
        return groups, group_index
    else:
        return groups


def flatten_list(array):
    return list(itertools.chain.from_iterable(array))


# =============================================================================
# Start of code
# =============================================================================
# Main code here
# Main code here
if __name__ == "__main__":
    # load data
    print('\n Loading and sorting data...')
    data = fits.getdata(DFILE1, ext=1)
    # ----------------------------------------------------------------------
    names = np.array(data['name'])
    # get number of peaks
    numpeaks = int((len(data.columns.names)-1)/4.0)
    # ----------------------------------------------------------------------
    # get a unique numeric name
    uids = get_numeric_id_from_name(names)
    # get the full region or sub region flag
    fullregion, subregion = get_subtype(names)
    # get groups
    gnames, gids, gmasks, glens = group_objects(names, uids)

    # ----------------------------------------------------------------------
    # construct data dict
    datadict = dict()
    datadict['uid'] = []
    datadict['group_name'] = []
    letters = string.ascii_uppercase
    for t_it in range(TOP_N):
        letter = letters[t_it]
        datadict['Period_{0}'.format(letter)] = []
        datadict['Power_{0}'.format(letter)] = []
        datadict['Spread_{0}'.format(letter)] = []

    # ----------------------------------------------------------------------
    # get top n cluster groups period, power and spread
    print('\n Getting clusters...')
    for uid in tqdm(uids):
        # get iteration parameters
        numregions = glens[uid]
        gmask = gmasks[uid]
        gname = gnames[uid][0]
        # if gname in datadict skip
        if gname.split('_Full')[0] in datadict['group_name']:
            continue
        # ----------------------------------------------------------------------
        # computation
        results = score_peaks(data, gmask, numpeaks, numregions, THRESHOLD)
        # ----------------------------------------------------------------------
        # add to data dict
        datadict['uid'].append(uid)
        datadict['group_name'].append(gname.split('_Full')[0])
        for t_it in range(TOP_N):
            letter = letters[t_it]
            datadict['Period_{0}'.format(letter)].append(results[0][t_it])
            datadict['Power_{0}'.format(letter)].append(results[1][t_it])
            datadict['Spread_{0}'.format(letter)].append(results[2][t_it])
    # ----------------------------------------------------------------------
    # save to table
    print('\n Saving data to file...')
    table = Table()
    for col in datadict:
        table[col] = datadict[col]
    table.write(DFILE2, format='fits', overwrite=True)

# =============================================================================
# End of code
# =============================================================================
