#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 12:38:38 2021
@author: zt
"""

import numpy as np


def dep_liberal_m2(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    total = np.nan

    dysphoric = (x.m2j12 == 1) and (x.m2j13 == 1 or x.m2j13 == 2 or x.m2j13 == 3) and (x.m2j13a == 1 or x.m2j13a == 2)
    anhedonia = (x.m2j14 == 1) and (x.m2j14a == 1 or x.m2j14a == 2 or x.m2j14a == 3) and (
                x.m2j14b == 1 or x.m2j14b == 2)

    not_dysphoric = (x.m2j12 == 2) or (x.m2j13 == 4) or (x.m2j13a == 3)
    not_anhedonia = (x.m2j14 == 2) or (x.m2j14a == 4) or (x.m2j14b == 3)

    if not_dysphoric or not_anhedonia:
        total = 0

    if dysphoric or anhedonia:
        total = 0
        bool_list = [x.m2j12 == 1, x.m2j13b == 1, x.m2j14 == 1, x.m2j15a == 1, x.m2j15b1 >= 10,
                     (x.m2j15c1 == 1 or x.m2j15c1 == 2), x.m2j15d == 1, x.m2j15e == 1, x.m2j15f == 1]
        for y in bool_list:
            if y:
                total += 1

    if x.m2j12 == -14 or x.m2j14 == -14 or x.m2j12 == 14 or x.m2j14 == 14 or x.m2j12 == -11 or x.m2j14 == -11:
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

def dep_liberal_f2(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    total = np.nan

    dysphoric = (x.f2j12 == 1) and (x.f2j13 == 1 or x.f2j13 == 2 or x.f2j13 == 3) and (x.f2j13a == 1 or x.f2j13a == 2)
    anhedonia = (x.f2j14 == 1) and (x.f2j14a == 1 or x.f2j14a == 2 or x.f2j14a == 3) and (
                x.f2j14b == 1 or x.f2j14b == 2)

    not_dysphoric = (x.f2j12 == 2) or (x.f2j13 == 4) or (x.f2j13a == 3)
    not_anhedonia = (x.f2j14 == 2) or (x.f2j14a == 4) or (x.f2j14b == 3)

    if not_dysphoric or not_anhedonia:
        total = 0

    if dysphoric or anhedonia:
        total = 0
        bool_list = [x.f2j12 == 1, x.f2j13b == 1, x.f2j14 == 1, x.f2j15a == 1, x.f2j15b1 >= 10,
                     (x.f2j15c1 == 1 or x.f2j15c1 == 2), x.f2j15d == 1, x.f2j15e == 1, x.f2j15f == 1]
        for y in bool_list:
            if y:
                total += 1

    if x.f2j12 == -14 or x.f2j14 == -14 or x.f2j12 == 14 or x.f2j14 == 14 or x.f2j12 == -11 or x.f2j14 == -11:
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


def emotionality_m2(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    lista = []
    listb = []

    if x.m2b17a < 0:
        lista.append(np.nan)
    else:
        lista.append(x.m2b17a)

    for i in [x.m2b17c, x.m2b17f]:  # manual reverse coding
        if i < 0:
            lista.append(np.nan)
        else:
            lista.append(6 - i)

    if x.m2b43a < 0:
        listb.append(np.nan)
    else:
        listb.append(x.m2b43a)

    for i in [x.m2b43c, x.m2b43f]:
        if i < 0:
            listb.append(np.nan)
        else:
            listb.append(6 - i)

    if lista.count(np.nan) > 0 or listb.count(np.nan) > 0:
        return np.nan

    if lista[0] >= 0:  # upon first positive value, return mean
        return np.mean(lista)

    if listb[0] >= 0:
        return np.mean(listb)

    return np.nan  # insurance


def emotionality_f2(x):
    """ 
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """

    lista = []
    listb = []

    if x.f2b16a < 0:
        lista.append(np.nan)
    else:
        lista.append(x.f2b16a)

    for i in [x.f2b16c, x.f2b16f]:
        if i < 0:
            lista.append(np.nan)
        else:
            lista.append(6 - i)

    if x.f2b37a < 0:
        listb.append(np.nan)
    else:
        listb.append(x.f2b37a)

    for i in [x.f2b37c, x.f2b37f]:
        if i < 0:
            listb.append(np.nan)
        else:
            listb.append(6 - i)

    if lista.count(np.nan) > 0 or listb.count(np.nan) > 0:
        return np.nan

    if lista[0] >= 0:  # upon first positive value, return mean
        return np.mean(lista)

    if listb[0] >= 0:
        return np.mean(listb)

    return np.nan  # insurance
