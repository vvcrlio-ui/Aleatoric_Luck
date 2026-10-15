#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 20:57:00 2021

@author: zt
"""
import numpy as np

# Note unit_conversion() requires a global variable, meta, which is the metadata object from pyreadstat's .read_dta()
# It also requires local variables UNIT_CONVERSION (dictionary) and FLIPPED_UNIT_CONVERSION

UNIT_CONVERSION = {
    1: ["1 Hour"],
    24: ["1 Days", "2 Day", "1 Day", "4 day(s)", "1 per day"],
    168: [
        "2 Weeks",
        "1 week",
        "1 Per week",
        "1 Week",
        "3 Week",
        "2 Week",
        "3 week(s)",
        "2 per week",
    ],
    336: [
        "2 Every 2 weeks",
        "2 Per 2 weeks",
        "2 2 weeks",
        "4 2 Weeks",
        "3 Two Weeks",
        "3 2 Weeks",
        "2 every 2 weeks",
        "3 per two weeks",
    ],
    504: ["101 Per 3 wks"],
    730.485: [
        "3 Months",
        "3 Month",
        "3 Per month",
        "4 Month",
        "3 month",
        "2 month(s)",
        "4 per month",
    ],
    2191.455: ["101 3 months"],
    4382.91: ["102 6 months"],
    8765.82: [
        "4 Years",
        "101 Per year",
        "102 Year",
        "103 Year",
        "1 year(s)",
    ],
    0: ["102 No payment req.", "102 Didn't pay", "0 none/payments legally terminated"],
}


def flip_dict(x):
    """
    Args: dict with values:[list of corresponding keys]
    Returns: dict with keys:values
    """
    new_dict = {}
    for value, key_list in x.items():
        for i in range(len(key_list)):
            new_dict[key_list[i]] = value
    return new_dict


FLIPPED_UNIT_CONVERSION = flip_dict(UNIT_CONVERSION)


def unit_standardization(x, meta):
    """
    Args: pd.Series where x.index[0] is question var name and x.index[1] is unit var name
    Output: float, unit standardized output in hours
    """
    question_var = str(x.index[0])
    unit_var = str(x.index[1])

    question = x[question_var]
    unit = x[unit_var]

    if question < 0:
        return np.nan

    unit_label = meta.variable_value_labels.get(unit_var).get(unit)
    # retrieves from metadata the long-form label e.g., "-9 Not in Wave" from "-9"

    # pulls out the key associated with the list containing the unit label
    hour_rate = FLIPPED_UNIT_CONVERSION.get(unit_label, np.nan)

    if hour_rate in [0, np.nan]:
        return hour_rate

    converted = question * hour_rate
    return converted


def unit_standardization_per(x, meta):
    """
    Use for type 2 problems, where it's usually "how many $ PER <unit>"
    Args: pd.Series where x.index[0] is question var name and x.index[1] is unit var name
    Output: float, unit standardized output in hours
    """
    question_var = str(x.index[0])
    unit_var = str(x.index[1])

    question = x[question_var]
    unit = x[unit_var]

    if question < 0:
        return np.nan

    unit_label = meta.variable_value_labels.get(unit_var).get(unit)
    # retrieves from metadata the long-form label e.g., "-9 Not in Wave" from "-9"

    # pulls out the key associated with the list containing the unit label
    hour_rate = FLIPPED_UNIT_CONVERSION.get(unit_label, np.nan)

    if hour_rate in [0, np.nan]:
        return hour_rate

    converted = question * (
        1 / hour_rate
    )  # this is the difference between this func and unit_standardization()
    return converted


def month_year(x, meta):
    """
    Args: pd.Series where x.index[0] is month var name and x.index[1] is year-date var name
    Output: float, output in months from 0 BCE
    """
    month_var = str(x.index[0])
    year_var = str(x.index[1])

    month = x[month_var]
    year = x[year_var]

    if month <= 0 or month > 12 or year < 0:
        return np.nan

    # make manual modifications for consistent output
    if (year_var == "r3f9_yr" or year_var == "d3g5_yr") and (100 > year > 0):
        year += 1900

    if year_var == "d3b1_yr" or year_var == "d3c1_yr":
        if 0 <= year < 6:
            year += 2000
        if 50 <= year < 100:
            year += 1900
        if 6 < year < 50:
            return np.nan

    new_date = year * 12 + month
    return new_date
