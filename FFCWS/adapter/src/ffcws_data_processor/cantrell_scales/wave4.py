#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 12:06:25 2021

@author: zt
"""

import numpy as np


def dep_liberal_m4(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    total = np.nan

    dysphoric = (x.m4j5 == 1) and (x.m4j6 == 1 or x.m4j6 == 2 or x.m4j6 == 3) and (x.m4j7 == 1 or x.m4j7 == 2)
    anhedonia = (x.m4j9 == 1) and (x.m4j10 == 1 or x.m4j10 == 2 or x.m4j10 == 3) and (x.m4j11 == 1 or x.m4j11 == 2)

    not_dysphoric = (x.m4j5 == 2) or (x.m4j6 == 4) or (x.m4j7 == 3)
    not_anhedonia = (x.m4j9 == 2) or (x.m4j10 == 4) or (x.m4j11 == 3)

    if not_dysphoric or not_anhedonia:
        total = 0

    if dysphoric or anhedonia:
        total = 0
        bool_list = [x.m4j5 == 1, x.m4j8 == 1, x.m4j9 == 1, x.m4j12 == 1, x.m4j13a >= 10,
                     (x.m4j14a == 1 or x.m4j14a == 2), x.m4j15 == 1, x.m4j16 == 1, x.m4j17 == 1]
        for y in bool_list:
            if y:
                total += 1

    if x.m4j5 == -14 or x.m4j9 == -14 or x.m4j5 == 14 or x.m4j9 == 14:
        total = np.nan

    caseness = {
        0: 0.0001,
        1: 0.0568,
        2: 0.2352,
        3: 0.5542,
        4: 0.8125,
        5: 0.8895,
        6: 0.8895,
        7: 0.9083,
        8: 0.9083
    }
    return caseness.get(total, np.nan)


def dep_liberal_f4(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    total = np.nan

    dysphoric = (x.f4j5 == 1) and (x.f4j6 == 1 or x.f4j6 == 2 or x.f4j6 == 3) and (x.f4j7 == 1 or x.f4j7 == 2)
    anhedonia = (x.f4j9 == 1) and (x.f4j10 == 1 or x.f4j10 == 2 or x.f4j10 == 3) and (x.f4j11 == 1 or x.f4j11 == 2)

    not_dysphoric = (x.f4j5 == 2) or (x.f4j6 == 4) or (x.f4j7 == 3)
    not_anhedonia = (x.f4j9 == 2) or (x.f4j10 == 4) or (x.f4j11 == 3)

    if not_dysphoric or not_anhedonia:
        total = 0

    if dysphoric or anhedonia:
        total = 0
        bool_list = [x.f4j5 == 1, x.f4j8 == 1, x.f4j9 == 1, x.f4j12 == 1, x.f4j13a >= 10,
                     (x.f4j14a == 1 or x.f4j14a == 2), x.f4j15 == 1, x.f4j16 == 1, x.f4j17 == 1]
        for y in bool_list:
            if y:
                total += 1

    if x.f4j5 == -14 or x.f4j9 == -14 or x.f4j5 == 14 or x.f4j9 == 14:
        total = np.nan

    caseness = {
        0: 0.0001,
        1: 0.0568,
        2: 0.2352,
        3: 0.5542,
        4: 0.8125,
        5: 0.8895,
        6: 0.8895,
        7: 0.9083,
        8: 0.9083
    }
    return caseness.get(total, np.nan)


def HOME_devStim4(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """
    total = []

    sublist = [x.p4c1a, x.p4c1b, x.p4c1c, x.p4c1d, x.p4c1e, x.p4c1f, x.p4c1g]
    for i in sublist:
        if i > 1:
            total.append(1)
        elif i == 1:
            total.append(0)
        else:
            total.append(np.nan)

    if x.p4c2 == 1:
        total.append(1)
    elif x.p4c2 == 0:
        total.append(0)
    else:
        total.append(np.nan)

    if x.p4c5 == 1:
        total.append(1)
    elif x.p4c5 == 0:
        total.append(0)
    else:
        total.append(np.nan)

    if x.p4c4 > 2:
        total.append(1)
    elif (x.p4c4 == 1 or x.p4c4 == 2 or x.p4c3 == 0):
        total.append(0)
    else:
        total.append(np.nan)

    if total.count(np.nan) > .2 * len(total):  # if greater than 20% NAs
        return np.nan  # return NA
    else:
        return np.nanmean(total)  # return mean ignoring NAs


def HOME_accessToReading4(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """
    total = []
    if x.p4c1h > 1:
        total.append(1)
    elif x.p4c1h == 1:
        total.append(0)
    else:
        total.append(np.nan)

    if x.p4c7 > 3:
        total.append(1)
    elif (x.p4c7 <= 3 and x.p4c7 > 0):
        total.append(0)
    else:
        total.append(np.nan)

    if x.p4c10 > 1:
        total.append(1)
    elif (x.p4c10 == 1 or x.p4c9 == 0 or x.p4c8 == 1):
        total.append(0)
    else:
        total.append(np.nan)

    if x.p4c12 > 2:
        total.append(1)
    elif (x.p4c12 == 1 or x.p4c12 == 2 or x.p4c11 == 0 or x.p4c9 == 0 or x.p4c8 == 1):
        total.append(0)
    else:
        total.append(np.nan)

    if total.count(np.nan) > .2 * len(total):  # if greater than 20% NAs
        return np.nan  # return NA
    else:
        return np.nanmean(total)  # return mean ignoring NAs


def HOME_outingActivities4(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """
    total = []

    if x.p4c16a >= 3:
        total.append(1)
    elif x.p4c16a > 0:
        total.append(0)
    else:
        total.append(np.nan)

    if x.p4c16b >= 3:
        total.append(1)
    elif x.p4c16b > 0:
        total.append(0)
    else:
        total.append(np.nan)

    if x.p4c16c >= 2:
        total.append(1)
    elif x.p4c16c > 0:
        total.append(0)
    else:
        total.append(np.nan)

    if x.p4c17a >= 2:
        total.append(1)
    elif x.p4c17a > 0:
        total.append(0)
    else:
        total.append(np.nan)

    if x.p4c17b >= 2:
        total.append(1)
    elif x.p4c17b > 0:
        total.append(0)
    else:
        total.append(np.nan)

    if x.p4c17c >= 2:
        total.append(1)
    elif x.p4c17c > 0:
        total.append(0)
    else:
        total.append(np.nan)

    if total.count(np.nan) > .2 * len(total):  # if greater than 20% NAs
        return np.nan  # return NA
    else:
        return np.nanmean(total)  # return mean ignoring NAs
