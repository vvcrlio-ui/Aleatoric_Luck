#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 12:24:05 2021

@author: zt
"""

import numpy as np


def dep_liberal_m3(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    total = np.nan

    dysphoric = (x.m3j5 == 1) and (x.m3j6 == 1 or x.m3j6 == 2 or x.m3j6 == 3) and (x.m3j7 == 1 or x.m3j7 == 2)
    anhedonia = (x.m3j9 == 1) and (x.m3j10 == 1 or x.m3j10 == 2 or x.m3j10 == 3) and (x.m3j11 == 1 or x.m3j11 == 2)

    not_dysphoric = (x.m3j5 == 2) or (x.m3j6 == 4) or (x.m3j7 == 3)
    not_anhedonia = (x.m3j9 == 2) or (x.m3j10 == 4) or (x.m3j11 == 3)

    if not_dysphoric or not_anhedonia:
        total = 0

    if dysphoric or anhedonia:
        total = 0
        bool_list = [x.m3j5 == 1, x.m3j8 == 1, x.m3j9 == 1, x.m3j12 == 1, x.m3j13a >= 10,
                     (x.m3j14a == 1 or x.m3j14a == 2), x.m3j15 == 1, x.m3j16 == 1, x.m3j17 == 1]
        for y in bool_list:
            if y:
                total += 1

    if x.m3j5 == -14 or x.m3j9 == -14 or x.m3j5 == 14 or x.m3j9 == 14:
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


def dep_liberal_f3(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    total = np.nan

    dysphoric = (x.f3j5 == 1) and (x.f3j6 == 1 or x.f3j6 == 2 or x.f3j6 == 3) and (x.f3j7 == 1 or x.f3j7 == 2)
    anhedonia = (x.f3j9 == 1) and (x.f3j10 == 1 or x.f3j10 == 2 or x.f3j10 == 3) and (x.f3j11 == 1 or x.f3j11 == 2)

    not_dysphoric = (x.f3j5 == 2) or (x.f3j6 == 4) or (x.f3j7 == 3)
    not_anhedonia = (x.f3j9 == 2) or (x.f3j10 == 4) or (x.f3j11 == 3)

    if not_dysphoric or not_anhedonia:
        total = 0

    if dysphoric or anhedonia:
        total = 0
        bool_list = [x.f3j5 == 1, x.f3j8 == 1, x.f3j9 == 1, x.f3j12 == 1, x.f3j14 >= 10, (x.f3j16 == 1 or x.f3j16 == 2),
                     x.f3j17 == 1, x.f3j18 == 1, x.f3j19 == 1]
        for y in bool_list:
            if y:
                total += 1

    if x.f3j5 == -14 or x.f3j9 == -14 or x.f3j5 == 14 or x.f3j9 == 14:
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


def alcohol_m3(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    caseness = {
        0: 0.0003,
        1: 0.0614,
        2: 0.3874,
        3: 0.8411,
        4: 1.0,
        5: 1.0,
        6: 1.0,
        7: 1.0
    }

    if x.m3j28 == 0 or x.m3j28 == 1:  # stem check
        return 0.0003

    if x.m3j28 < 0:  # stem check
        return np.nan

    total = 0
    for y in [x.m3j29, x.m3j30, x.m3j31, x.m3j32, x.m3j33, x.m3j34, x.m3j35]:  # count symptoms
        if y == 1:
            total += 1
        if y == 3:
            return 0.0003

    return caseness.get(total, np.nan)


def alcohol_f3(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """
    caseness = {
        0: 0.0003,
        1: 0.0614,
        2: 0.3874,
        3: 0.8411,
        4: 1.0,
        5: 1.0,
        6: 1.0,
        7: 1.0
    }

    if x.f3j33 == 0 or x.f3j33 == 1:
        return 0.0003

    if x.f3j33 < 0:
        return np.nan

    total = 0
    for y in [x.f3j35, x.f3j37, x.f3j38, x.f3j39, x.f3j40, x.f3j41, x.f3j43]:
        if y == 1:
            total += 1
        if y == 3:
            return 0.0003

    return caseness.get(total, np.nan)


def drug_m3(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    drug_use = [x.m3j36a, x.m3j36b, x.m3j36c, x.m3j36d,
                x.m3j36e, x.m3j36f, x.m3j36g, x.m3j36h,
                x.m3j36i]
    check_symptom = 0

    drug_na_count = len(drug_use) - len([i for i in drug_use if i >= 0])

    if drug_use.count(1) > 0:
        check_symptom = 1

    if drug_na_count == len(drug_use):
        check_symptom = np.nan

    if x.m3j36j == 1:
        check_symptom = 1

    if check_symptom == 0:
        return 0

    if np.isnan(check_symptom):
        return np.nan

    symptom = [x.m3j37, x.m3j38, x.m3j39, x.m3j40, x.m3j41, x.m3j42, x.m3j43]
    total = symptom.count(1)

    caseness = {
        0: 0.0000,
        1: 0.0492,
        2: 0.2787,
        3: 0.7561,
        4: 1.0,
        5: 1.0,
        6: 1.0,
        7: 1.0
    }
    return caseness.get(total, np.nan)


def drug_f3(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    drug_use = [x.f3j44a, x.f3j44b, x.f3j44c, x.f3j44d,
                x.f3j44e, x.f3j44f, x.f3j44g, x.f3j44h,
                x.f3j44i]

    check_symptom = 0

    drug_na_count = len(drug_use) - len([i for i in drug_use if i >= 0])

    if drug_use.count(1) > 0:
        check_symptom = 1

    if drug_na_count == len(drug_use):
        check_symptom = np.nan

    if x.f3j44j == 1:
        check_symptom = 1

    if check_symptom == 0:
        return 0

    if np.isnan(check_symptom):
        return np.nan

    symptom = [x.f3j45, x.f3j46, x.f3j47, x.f3j48, x.f3j49, x.f3j50, x.f3j51]
    total = symptom.count(1)

    caseness = {
        0: 0.0000,
        1: 0.0492,
        2: 0.2787,
        3: 0.7561,
        4: 1.0,
        5: 1.0,
        6: 1.0,
        7: 1.0
    }
    return caseness.get(total, np.nan)


def HOME_devStim3(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    binary = []
    for i in x:
        if i > 1:
            binary.append(1)
        elif i == 1:
            binary.append(0)
        else:
            binary.append(np.nan)

    if binary.count(np.nan) > .2 * len(binary):  # use 80% non-NA rule
        return np.nan  # return NA
    else:
        return np.nanmean(binary)  # return mean ignoring NAs
