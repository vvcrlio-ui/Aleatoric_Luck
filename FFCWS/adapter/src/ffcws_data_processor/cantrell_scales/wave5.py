#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:59:58 2021

@author: zt
"""

import numpy as np


def sdq(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    nonNA_list = []

    for val in x:
        if type(val) is int or type(val) is float:
            if val >= 0:
                nonNA_list.append(val)

    if len(nonNA_list) >= 10:
        return sum(nonNA_list) / len(nonNA_list)

    return np.nan


def sdq_externalizing(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    nonNA_list = []

    for val in x:
        if type(val) is int or type(val) is float:
            if val >= 0:
                nonNA_list.append(val)

    if len(nonNA_list) >= 4:
        return sum(nonNA_list) / len(nonNA_list)

    return np.nan


def sdq_internalizing(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    nonNA_list = []

    for val in x:
        if type(val) is int or type(val) is float:
            if val >= 0:
                nonNA_list.append(val)

    if len(nonNA_list) >= 6:
        return sum(nonNA_list) / len(nonNA_list)

    return np.nan


def dep_liberal_m5(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    total = np.nan

    dysphoric = (x.m5g3 == 1) and (x.m5g4 == 1 or x.m5g4 == 2 or x.m5g4 == 3) and (x.m5g5 == 1 or x.m5g5 == 2)
    anhedonia = (x.m5g7 == 1) and (x.m5g8 == 1 or x.m5g8 == 2 or x.m5g8 == 3) and (x.m5g9 == 1 or x.m5g9 == 2)

    not_dysphoric = (x.m5g3 == 2) or (x.m5g4 == 4) or (x.m5g5 == 3)
    not_anhedonia = (x.m5g7 == 2) or (x.m5g8 == 4) or (x.m5g9 == 3)

    if (not_dysphoric or not_anhedonia):
        total = 0

    if dysphoric or anhedonia:
        total = 0
        bool_list = [x.m5g3 == 1, x.m5g6 == 1, x.m5g7 == 1, x.m5g10 == 1, x.m5g11a >= 10,
                     (x.m5g12a == 1 or x.m5g12a == 2), x.m5g13 == 1, x.m5g14 == 1, x.m5g15 == 1]
        for y in bool_list:
            if y:
                total += 1

    if x.m5g3 == -14 or x.m5g7 == -14 or x.m5g3 == 14 or x.m5g7 == 14:
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


def dep_liberal_f5(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    total = np.nan

    dysphoric = (x.f5g3 == 1) and (x.f5g4 == 1 or x.f5g4 == 2 or x.f5g4 == 3) and (x.f5g5 == 1 or x.f5g5 == 2)
    anhedonia = (x.f5g7 == 1) and (x.f5g8 == 1 or x.f5g8 == 2 or x.f5g8 == 3) and (x.f5g9 == 1 or x.f5g9 == 2)

    not_dysphoric = (x.f5g3 == 2) or (x.f5g4 == 4) or (x.f5g5 == 3)
    not_anhedonia = (x.f5g7 == 2) or (x.f5g8 == 4) or (x.f5g9 == 3)

    if not_dysphoric or not_anhedonia:
        total = 0

    if dysphoric or anhedonia:
        total = 0
        for y in [x.f5g3 == 1, x.f5g6 == 1, x.f5g7 == 1, x.f5g10 == 1, x.f5g11a >= 10,
                  (x.f5g12a == 1 or x.f5g12a == 2), x.f5g13 == 1, x.f5g14 == 1, x.f5g15 == 1]:
            if y:
                total += 1

    if x.f5g3 == -14 or x.f5g7 == -14 or x.f5g3 == 14 or x.f5g7 == 14:
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


def dep_liberal_n5(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    total = np.nan

    dysphoric = (x.n5f3 == 1) and (x.n5f4 == 1 or x.n5f4 == 2 or x.n5f4 == 3) and (x.n5f5 == 1 or x.n5f5 == 2)
    anhedonia = (x.n5f7 == 1) and (x.n5f8 == 1 or x.n5f8 == 2 or x.n5f8 == 3) and (x.n5f9 == 1 or x.n5f9 == 2)

    not_dysphoric = (x.n5f3 == 2) or (x.n5f4 == 4) or (x.n5f5 == 3)
    not_anhedonia = (x.n5f7 == 2) or (x.n5f8 == 4) or (x.n5f9 == 3)

    if not_dysphoric or not_anhedonia:
        total = 0

    if dysphoric or anhedonia:
        total = 0
        for y in [x.n5f3 == 1, x.n5f6 == 1, x.n5f7 == 1, x.n5f10 == 1, x.n5f11a >= 10,
                  (x.n5f12a == 1 or x.n5f12a == 2), x.n5f13 == 1, x.n5f14 == 1, x.n5f15 == 1]:
            if y:
                total += 1

    if x.n5f3 == -14 or x.n5f7 == -14 or x.n5f3 == 14 or x.n5f7 == 14:
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


def dep_conservative_n5(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    dysphoric = (x.n5f3 == 1) and (x.n5f4 == 1 or x.n5f4 == 2) and (x.n5f5 == 1 or x.n5f5 == 2)
    anhedonia = (x.n5f7 == 1) and (x.n5f8 == 1 or x.n5f8 == 2) and (x.n5f9 == 1 or x.n5f9 == 2)

    total = np.nan

    not_dysphoric = (x.n5f3 == 2) or (x.n5f4 == 4 or x.n5f4 == 3) or (x.n5f5 == 3)
    not_anhedonia = (x.n5f7 == 2) or (x.n5f8 == 4 or x.n5f8 == 3) or (x.n5f9 == 3)

    if not_dysphoric or not_anhedonia:
        total = 0

    if dysphoric or anhedonia:
        total = 0
        for y in [x.n5f6 == 1, x.n5f7 == 1, x.n5f10 == 1, x.n5f11a >= 10,
                  (x.n5f12a == 1 or x.n5f12a == 2), x.n5f13 == 1, x.n5f14 == 1, x.n5f15 == 1]:
            if y:
                total += 1

    if x.n5f3 == -14 or x.n5f7 == -14 or x.n5f3 == 14 or x.n5f7 == 14:
        total = np.nan

    caseness = {
        0: 0.0001,
        1: 0.0568,
        2: 0.2352,
        3: 0.5542,
        4: 0.8125,
        5: 0.8895,
        6: 0.8895,
        7: 0.9083
    }
    return caseness.get(total, np.nan)


def pubertal_p5(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    if x.p5h22 == 1:
        x.p5h22 = 4
    if x.p5h22 == 2:
        x.p5h22 = 1

    count = 0
    total = 0
    for i in x:
        if 1 <= i <= 4:
            count += 1
            total += i

    if count != 5:
        return np.nan

    return total / count
