#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/04/17 at 4:17 PM

@author: neil

Program description here

Version 0.0.0
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from tqdm import tqdm
import string
import itertools


# =============================================================================
# Define variables
# =============================================================================
WORKSPACE = '/Astro/Projects/RayPaul_Work/SuperWASP/'
DPATH = WORKSPACE + 'Data/ls_analysis_run/'
DFILE1 = DPATH + '/light_curve_analysis_periods_regions.fits'
DFILE2 = DPATH + '/light_curve_analysis_periods_regions_tmp.fits'

# -----------------------------------------------------------------------------
THRESHOLD = 10.0


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


def peak_in_peak_old(group1, group2, threshold):
    # get the combinations of the peaks
    combinations = np.array(list(itertools.product(group1, group2)))
    # work out how different peaks are
    differences = abs(combinations[:, 0] - combinations[:, 1])
    # work out the fractional difference to the group1 peak
    ndifferences = differences/combinations[:, 0]
    # we want those peaks greater than zero (not identical) but less than the
    # threshold percentage
    mask = (ndifferences > 0) & (ndifferences <= threshold/100.0)

    return combinations[mask]


def peak_in_peak(test, array, threshold, include_point=False):
    # work out how different peaks are
    differences = abs(array - test)
    # work out the fractional difference to the group1 peak
    ndifferences = differences/test
    # we want those peaks greater than zero (not identical) but less than the
    # threshold percentage
    if include_point:
        mask = (ndifferences <= threshold / 100.0)
    else:
        mask = (ndifferences > 0) & (ndifferences <= threshold/100.0)
    # return mask
    return mask


def weighted_peak_value(peaks, w, axis=None):
    if axis is None:
        weighted_peak = np.nansum(w*peaks)/np.nansum(w)
    else:
        weighted_peak = np.nansum(w*peaks, axis=axis)/np.nansum(w, axis=axis)
    return weighted_peak


def score_peaks_old(gnames, gmasks, glens, numpeaks, threshold):
    ratings, nratings = dict(), dict()
    for group in tqdm(gnames):
        # score:
        # best  peak1 is in both the full and one or more regions
        #       and not in any noise peak
        scores = np.zeros(numpeaks)
        nscores = np.zeros(numpeaks)
        for n in range(numpeaks):
            # get all peals for Peak{N} for this group
            peaks = np.array(data['Peak{0}'.format(n+1)])[gmasks[group]]
            npeaks = np.array(data['nPeak{0}'.format(n+1)])[gmasks[group]]
            # get good peaks
            good_peaks = peak_in_peak(peaks, peaks, threshold)
            bad_peaks = peak_in_peak(peaks, npeaks, threshold)
            # check if we have any good peaks
            # if len(good_peaks) > 0:
                # score should go down with a lesser peak
                # scores[n] += 1000 * (numpeaks - n)
            # Check for bad peaks and good peak matches
            cond1s, cond2s = [], []
            # loop around each group member
            for p, pp in enumerate(peaks):
                # check that peak is in a good peak pair
                if len(good_peaks) > 0:
                    cond1 = pp in good_peaks[:, 0]
                else:
                    cond1 = False
                # check that peak is NOT in a bad peak pair
                if len(bad_peaks) > 0:
                    cond2 = pp not in bad_peaks[:, 0]
                # if bad peak pairs are empty then peak is definitely NOt in a bad
                # peak pair
                else:
                    cond2 = True
                cond1s.append(cond1)
                cond2s.append(cond2)
            # add 100 * number of matches of good peak
            scores[n] += 100 * np.sum(cond1s)
            # add 10 * number of matches of not bad peak
            scores[n] += 10 * np.sum(cond2s)
            # normalise scores by number in group
            nscores[n] = scores[n] / glens[group]
        # add scores to group ratings
        ratings[group] = scores
        nratings[group] = nscores
    return ratings, nratings


def score_peaks_old3(gnames, gmasks, glens, numpeaks, threshold):
    ratings, wperiods, groupstats = dict(), dict(), dict()
    for group in tqdm(gnames):
        # ----------------------------------------------------------------------
        # number of regions
        numregions = glens[group]
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
            peakm = np.array(data['Peak{0}'.format(m+ 1)])[gmasks[group]]
            peaks.append(peakm)
            powerm = np.array(data['Power{0}'.format(m+1)])[gmasks[group]]
            powers.append(powerm)
            npeakm = np.array(data['nPeak{0}'.format(m + 1)])[gmasks[group]]
            npeaks.append(npeakm)
            npowerm = np.array(data['nPower{0}'.format(m+1)])[gmasks[group]]
            npowers.append(npowerm)
        peaks, npeaks = np.array(peaks), np.array(npeaks)
        powers, npowers = np.array(powers), np.array(npowers)
        # ----------------------------------------------------------------------
        #normalise the powers to the largest peak in each region
        norm_power = np.zeros_like(powers)
        for m in range(numpeaks):
            for n in range(numregions):
                norm_power[m, n] = powers[m, n]/np.nanmax(powers[:, n])
        # ----------------------------------------------------------------------
        # use the npeaks to rule out some peaks as true peaks
        # define a true peak to appear in at least one other region
        keep_peak = np.zeros(peaks.shape, dtype=bool)
        for m in range(numpeaks):
            for n in range(numregions):
                testpeak = peaks[m, n]
                # get good peaks and bad peaks
                good_peaks = peak_in_peak(testpeak, peaks, threshold)
                bad_peaks = peak_in_peak(testpeak, npeaks, threshold)
                # condition 1: no good peak is near a bad peak
                c1 = np.sum(bad_peaks) == 0
                # condition 2: we have at least another good_peak detection
                numgood = np.sum(good_peaks[:, :m-1])
                numgood += np.sum(good_peaks[:, m+1:])
                c2 = numgood > 1
                # if both conditions met then keep peak as possible true peak
                if c1 and c2:
                    keep_peak[m, n] = True
        # all peaks left will have no be near any peak in "npeaks" and will
        # appear at least twice in "peaks"
        # ----------------------------------------------------------------------
        # define a weight based on the power of the peak
        # if we aren't keeping the peak set the power to 0
        weight = keep_peak * norm_power**2



        # find clusters of peaks
        peak_clusters = cluster((peaks*keep_peak).flat, percentagemaxgap=threshold)
        cmedians, cstds = [], []
        for peak_cluster in peak_clusters:
            cmedians = np.append(cmedians, np.nanmedian(peak_cluster))
            cstds = np.append(cstds, np.nanstd(peak_cluster))

        medians = np.zeros_like(peaks)
        stds = np.zeros_like(peaks)
        nums = np.zeros_like(peaks)
        for m in range(numpeaks):
            for n in range(numregions):
                testpeak = peaks[m, n]
                clustergroup = peak_in_peak(testpeak, cmedians, threshold,
                                            True)
                if np.sum(clustergroup) > 0:
                    medians[m, n] = cmedians[clustergroup]
                    stds[m, n] = cstds[clustergroup]
        # work out the number of times a median occurs
        for m in range(numpeaks):
            for n in range(numregions):
                nums[m, n] = np.sum(np.in1d(medians, medians[m, n]))

        weight = weight * (nums/nums.size)**2
        ratings[group] = weight
    # return ratings
    return ratings, wperiods, groupstats


def score_peaks(gnames, gmasks, glens, numpeaks, threshold):
    ratings, targetperiods, groupperiods = dict(), dict(), dict()
    for group in tqdm(gnames):
        # ----------------------------------------------------------------------
        # number of regions
        numregions = glens[group]
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
            peakm = np.array(data['Peak{0}'.format(m+ 1)])[gmasks[group]]
            peaks.append(peakm)
            powerm = np.array(data['Power{0}'.format(m+1)])[gmasks[group]]
            powers.append(powerm)
            npeakm = np.array(data['nPeak{0}'.format(m + 1)])[gmasks[group]]
            npeaks.append(npeakm)
            npowerm = np.array(data['nPower{0}'.format(m+1)])[gmasks[group]]
            npowers.append(npowerm)
        peaks, npeaks = np.array(peaks), np.array(npeaks)
        powers, npowers = np.array(powers), np.array(npowers)
        # ----------------------------------------------------------------------
        # use the npeaks to rule out some peaks as true peaks
        # define a true peak to appear in at least one other region
        keep_peak = np.zeros(peaks.shape, dtype=bool)
        for m in range(numpeaks):
            for n in range(numregions):
                testpeak = peaks[m, n]
                # get good peaks and bad peaks
                good_peaks = peak_in_peak(testpeak, peaks, threshold)
                bad_peaks = peak_in_peak(testpeak, npeaks, threshold)
                # condition 1: no good peak is near a bad peak
                c1 = np.sum(bad_peaks) == 0
                # condition 2: we have at least another good_peak detection
                numgood = np.sum(good_peaks[:, :m-1])
                numgood += np.sum(good_peaks[:, m+1:])
                c2 = numgood > 1
                # if both conditions met then keep peak as possible true peak
                if c1 and c2:
                    keep_peak[m, n] = True
        # ----------------------------------------------------------------------
        peaks = keep_peak * peaks
        powers = keep_peak * powers
        weights = np.zeros_like(powers)

        # all peaks left will have no be near any peak in "npeaks" and will
        # appear at least twice in "peaks"
        flatpeaks = peaks.flatten()
        flatx = np.indices(keep_peak.shape)[0].flatten()
        flaty = np.indices(keep_peak.shape)[1].flatten()
        peak_cg, cgi = cluster(flatpeaks, percentagemaxgap=threshold,
                          return_indices=True)
        # get the powers associated with each cluster group (from original
        # positions power array
        power_cg, cgroup_powers = [], []
        # loop around each group
        for i in range(len(cgi)):
            power_cg.append([])
            # loop around the members in this group
            for j in range(len(cgi[i])):
                pos = cgi[i][j]
                power_cg[i].append(powers[flatx[pos], flaty[pos]])
        # get the sum of the powers for each cluster group
        for i in range(len(cgi)):
            cgroup_powers.append(np.nansum(power_cg[i]))
        # the weights are equal to the sum of the powers in a group
        # divided by the total power
        for i in range(len(cgi)):
            for j in range(len(cgi[i])):
                pos = cgi[i][j]
                value = cgroup_powers[i]/np.nansum(cgroup_powers)
                weights[flatx[pos], flaty[pos]] = value
        ratings[group] = weights

        targetperiod = weighted_peak_value(peaks, weights, axis=0)
        groupperiod = weighted_peak_value(peaks, weights)

        targetperiods[group] = targetperiod
        groupperiods[group] = groupperiod
    return ratings, targetperiods, groupperiods


def cluster(data, maxgap=None, percentagemaxgap=None, return_indices=False):
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
        raise ValueError("Must define either maxgap or percentagemaxgap")

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


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # load data
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
    # possible good periods
    ratings, tps, gps = score_peaks(gnames, gmasks, glens, numpeaks, THRESHOLD)
    # add rating to array
    rating_arr = np.zeros((len(uids), numpeaks), dtype=float)
    targetp_arr = np.zeros(len(uids))

    for it, uid in enumerate(uids):
        pos = np.where(names[it] == np.array(gnames[uids[it]]))[0][0]
        rating_arr[it] = ratings[uid].T[pos]
        targetp_arr[it] = tps[uid][pos]
    # add group length to array
    grouplen_arr, gperiod = [], []
    for it, uid in enumerate(uids):
        grouplen_arr.append(glens[uid])
        gperiod.append(gps[uid])
    # ----------------------------------------------------------------------
    table = Table()
    table['uid'] = uids
    table['Full'] = fullregion
    table['Sub'] = subregion
    table['Ngroup'] = grouplen_arr
    table['Object_period'] = targetp_arr
    table['Group_period'] = gperiod
    # table['Group_e_period'] = geperiod
    for col in data.columns.names:
        table[col] = data[col]
        if "Peak" in col and "nPeak" not in col:
            peaknum = int(col.split("Peak")[1])
            # Add score column for each peak
            newcol = 'Score{0}'.format(peaknum)
            table[newcol] = rating_arr[:, peaknum-1]

    table.write(DFILE2, format='fits', overwrite=True)



# =============================================================================
# End of code
# =============================================================================
