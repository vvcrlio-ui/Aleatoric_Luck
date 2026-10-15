# flake8: noqa
#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 15:13:01 2021
@author: zt
@author: Pranay Anchuri <2022-05-20 Fri>
"""

import numpy as np
import pandas as pd
import math


def Home_homeInterior(data):
    """
    Items are o3r[1-10], o3r[12-13]
    Original values are 0:1
    Set entries outside original values to NA
    Reverse code o3r1, o3r[3-10], o3r[12-13]
    Take the mean of the reverse-coded items and the o3r2 item as long as 80% of these items are non-missing
    """
    combined = []
    for x in data:
        combined.extend(x if isinstance(x, list) else [x])

    combined = pd.Series(combined)

    def mean_80pct(x):
        return x.mean() if x.count() / x.size >= 0.8 else np.nan

    return mean_80pct(combined)


# custom
def reverseSubset_andsubOne_mean(x):
    """
    Relevant for when you have to reverse a subset of a scale and subtract one from all items.
    Args: x (pd.Series) containing intermediate (already reversed) list and other items
    Output: (float) scale
    """

    intermediate_list = []
    total = []

    for index, item in zip(x.index, x):
        if "intermediate" in index:
            intermediate_list = item
        elif item < 0:
            total.append(np.nan)
        else:
            total.append(item - 1)  # subtract one from all items

    for i in intermediate_list:  # subtract one from all items
        i -= 1
        total.append(i)

    return sum(total) / len(total)


def reverseSubset_andsubOne_sum(x):
    """
    Relevant for when you have to reverse a subset of a scale and subtract one from all items.
    Args: x (pd.Series) containing intermediate (already reversed) list and other items
    Output: (float) scale
    """

    intermediate_list = []
    total = []

    for index, item in zip(x.index, x):
        if "intermediate" in index:
            intermediate_list = item
        elif item < 0:
            total.append(np.nan)
        else:
            total.append(item - 1)  # subtract one from all items

    for i in intermediate_list:  # subtract one from all items
        i -= 1
        total.append(i)

    return sum(total)


# custom
def conflict_tactics(x):
    """
    Args: x (pd.Series) = variable values for scale; note, could be rewritten in dictionary form
    output: (float) scale
    """

    li = []
    for i in x:
        if i == 0:
            li.append(0)
        elif i == 1:
            li.append(1)
        elif i == 2:
            li.append(2)
        elif i == 3:
            li.append(4)
        elif i == 4:
            li.append(8)
        elif i == 5:
            li.append(15)
        elif i == 6:
            li.append(25)
        elif i == 7:
            li.append(0)
        elif i == 8:
            li.append(0)
        else:
            return np.nan
    return sum(li) / len(li)


def familyMentalHealth(x):
    """
    Args: x (pd.Series) = contains two lists of values for wave 3 and wave 4
    output: (float) scale
    """

    # NOTE: familyMentalHealth(), CBCL_combo(), list_mean(), and list_mean_80pct() can potentially be consolidated
    # Return the non-missing value of the two intermediate scales, if any
    first = math.nan
    second = math.nan
    for index, item in zip(x.index, x):
        if "intermediate1" in index:
            first = item
        elif "intermediate2" in index:
            second = item
    if not math.isnan(first):
        return first
    elif not math.isnan(second):
        return second
    else:
        return np.nan


# custom
def CBCL_combo(x):
    """
    Args: x (pd.Series) = contains two lists of values for res and non-res, plus other vars
    Note that the res and non-res lists should be parallel (i.e., x[0] relates to y[0])
    output: (float) scale
    """

    res = []
    non_res = []
    combined = []

    for index, item in zip(x.index, x):
        if "_nonres" in index:
            non_res = item
        elif "_res" in index:
            res = item
        elif item < 0:
            combined.append(np.nan)
        else:
            combined.append(item)

    # add res or non-res list, depending on which applies
    for i in range(len(res)):
        if np.isnan(res[i]) and np.isnan(non_res[i]):
            combined.append(np.nan)  # if both na, add na
            continue
        if res[i] >= 0:  # upon first positive value, append the rest of that list
            combined.extend(res[i:])
            break
        if non_res[i] >= 0:
            combined.extend(non_res[i:])
            break

    non_na_list = [i for i in combined if i >= 0]

    if len(non_na_list) / len(combined) >= 0.8:  # apply 80% non-NA rule
        return sum(non_na_list) / len(non_na_list)
    else:
        return np.nan


# custom
def list_mean(x):
    """
    Args: x (pd.Series) = contains one list (e.g., corresponding to non/res) and other vars
    e.g., for the complement non/res. Lists should be in same order; does not implement 80% non-NA rule
    output: (float) scale
    """

    lista = []
    listb = []
    for index, item in zip(x.index, x):
        if "_intermediate" in index:
            lista = item
        elif item < 0:
            listb.append(np.nan)
        else:
            listb.append(item)

        # add lista or listb, depending on which applies
    if lista.count(np.nan) > 0 and listb.count(np.nan) > 0:
        return np.nan  # if both na, return na

    if lista[0] >= 0:  # upon first positive value, use that list
        return np.mean(lista)

    if listb[0] >= 0:  # upon first positive value, use that list
        return np.mean(listb)


# custom
def list_mean_80pct(x):
    """
    Args: x (pd.Series) = contains one list (e.g., corresponding to non/res) and other vars
    e.g., for the complement non/res. Lists should be in same order; uses the 80% non-NA rule
    output: (float) scale
    """

    lista = []
    listb = []
    combined = []
    for index, item in zip(x.index, x):
        if "_intermediate" in index:
            lista = item
        elif item < 0:
            listb.append(np.nan)
        else:
            listb.append(item)

    for i in range(len(lista)):
        if np.isnan(lista[i]) and np.isnan(listb[i]):
            combined.append(np.nan)  # if both na, return na
            continue

        if lista[i] >= 0:  # upon first positive value, append the rest of that list
            combined.extend(lista[i:])
            break

        if listb[i] >= 0:  # upon first positive value, append the rest of that list
            combined.extend(listb[i:])
            break

    non_na_list = [i for i in combined if i >= 0]

    if len(non_na_list) / len(combined) >= 0.8:
        return sum(non_na_list) / len(non_na_list)
    else:
        return np.nan


# custom
# custom
def food_insecurity(x):
    """
    Args: x (pd.Series) = variable values for scale
    output: (float) scale
    """
    changed = []
    generic_index = (
        []
    )  # a list with generic identifiers w same order as x (allows this to work for both wave 3 and 4
    # a similar format could be used to generalize depression scales, but it would have to rely on order, not suffixes

    for i in x.index:
        generic_index.append(str(i)[2:])

    imputation_order = [
        "d1a",
        "d1b",
        "d1d",
        "d1c",
        "d1e",
        "d4",
        "d5",
        "d4a",
        "d3",
        "d6",
        "d7",
        "d10",
        "d9",
        "d12",
        "d9a",
        "d11",
        "d11a",
        "d13",
    ]

    # convert scales from 1-3 (1 and 2 endorsed, 3 not endorsed) to 0-1 (non-end, end)
    for i in range(x.size):
        if x[i] == 1 or x[i] == 2:
            x[i] = 1
        if x[i] == 3:
            x[i] = 0

    # first, impute individual questions
    for i in [["d4a", "d4"], ["d9a", "d9"], ["d11a", "d11"], ["d13", "d12"]]:
        # impute i[0] to 0 if i[0] is missing and if i[1]==0
        if x[generic_index.index(i[0])] < 0:
            if x[generic_index.index(i[1])] == 0:
                x[generic_index.index(i[0])] = 0

    # now impute d2 (screener question)
    x[generic_index.index("d2")] = 0
    d2_zeroes = 0
    for i in ["d1a", "d1b", "d1c", "d1d", "d1e"]:
        val = x[generic_index.index(i)]
        if val < 0:
            val = np.nan
        changed.append(
            [i, val]
        )  # add these to the scale list, but no need to impute anything here

        if val == 1:
            x[generic_index.index("d2")] = 1
        if val == 0:
            d2_zeroes += 1

    if d2_zeroes < 4 and x[generic_index.index("d2")] == 0:
        x[
            generic_index.index("d2")
        ] = -1  # acts as NA, can't be NA due to needing consistent types(?)

    # impute questions depending on d2
    for i in ["d3", "d4", "d4a", "d5", "d6", "d7"]:
        indice = generic_index.index(i)
        orig_val = x[indice]

        if orig_val < 0:
            if x[generic_index.index("d2")] == 0:
                orig_val = 0
                x[indice] = 0
            else:
                orig_val = np.nan
                x[indice] = -1
        changed.append([i, orig_val])

    # now impute d8
    x[generic_index.index("d8")] = 0
    d8_zeroes = 0
    for i in ["d3", "d4", "d4a", "d5", "d6", "d7"]:
        val = x[generic_index.index(i)]

        if val == 1:
            x[generic_index.index("d8")] = 1
        if val == 0:
            d8_zeroes += 1

    if x[generic_index.index("d8")] == 0 and d8_zeroes < 5:
        x[generic_index.index("d8")] = -1  # acts as NA

    # impute questions depending on d8
    for i in ["d9", "d9a", "d10", "d11", "d11a", "d12", "d13"]:
        indice = generic_index.index(i)
        orig_val = x[indice]

        if orig_val < 0:
            if x[generic_index.index("d8")] == 0:
                orig_val = 0
            else:
                orig_val = np.nan
        changed.append([i, orig_val])
    # changed format is: [generic_index, orig_val]

    # kick out if more than 3 NAs
    countNA = 0
    for i in changed:
        if np.isnan(i[1]):
            countNA += 1
    if countNA > 3:
        return np.nan

    # impute here based on severity scale
    for z in range(len(changed)):
        j = changed[z]

        if not np.isnan(j[1]):  # we only care about NA values to impute
            continue

        index_val = imputation_order.index(
            j[0]
        )  # index val within the imputation order
        has_neg = 0  # does this val have negative responses to the left of it in the imputation order?
        has_pos = 0  # does this val have positive responses to the right of it in the imputation order?

        # find if there is a negative response before this item
        for k in imputation_order[:index_val]:

            if has_neg == 1:
                break

            if (
                index_val == 0
            ):  # if first val in imputation order, can assume negative response to left
                break

            for (
                l
            ) in (
                changed
            ):  # have to find the match to k; could use dict instead for speed?
                if l[0] == k:
                    if l[1] == 0:  # is there is a negative response before the item
                        changed[z][1] = 0  # if there is, impute item as 0 and move on
                        has_neg = 1
                        break

        if has_neg == 1:
            continue
            # if no previous negative val, look for a pos answer after it in the imputation order
        for m in imputation_order[index_val + 1 :]:
            if has_pos == 1:
                break

            if (
                index_val == len(imputation_order) - 1
            ):  # if last item, can assume pos value exists after and impute 1
                changed[z][1] = 1
                has_pos = 1
                break

            for n in changed:
                if n[0] == m:
                    if n[1] == 1:
                        changed[z][1] = 1  # if there is a pos item after, impute as 1
                        has_pos = 1
                        break
        if has_pos == 0:
            changed[z][1] = 0  # if no pos match, impute 0

    total = 0
    for i in changed:
        total += i[1]

    final_conversions = {
        0: 0,
        1: 1,
        2: 1.8,
        3: 2.4,
        4: 3,
        5: 3.4,
        6: 3.9,
        7: 4.3,
        8: 4.7,
        9: 5.1,
        10: 5.5,
        11: 5.9,
        12: 6.3,
        13: 6.6,
        14: 7,
        15: 7.4,
        16: 8,
        17: 8.7,
        18: 9.3,
    }

    return final_conversions.get(total, np.nan)
