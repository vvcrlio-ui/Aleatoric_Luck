# Adapted from scales/recode.py of emilycantrell/predicting-life-outcomes-ffcws
# (commit 995c845, Apache License 2.0). Changes: swifter replaced by
# DataFrame.apply, the wave 6 scales and the file readers removed, and
# compute_scales added to run only the requested instruction rows.

import re
from inspect import signature

import numpy as np
import pandas as pd

from .custom_scales import *
from .hetero_units import *
from .wave2 import *
from .wave3 import *
from .wave4 import *
from .wave5 import *


def get_range_str(s):
    try:
        return re.search(r"\[(.*)\]", s).group(1)
    except AttributeError:
        return None


def get_var_list(varstr):
    """Expand e.g. "m2j[1-3], m2k" into ["m2j1", "m2j2", "m2j3", "m2k"]."""
    csplit = varstr.split(",")
    vars = []
    for token in csplit:
        token = token.strip()
        r = get_range_str(token)
        if r is None:
            vars.append(token)
        else:
            prefix = token.split("[")[0]
            start, end = r.split("-")
            try:
                vars.extend(
                    ["%s%d" % (prefix, x) for x in range(int(start), int(end) + 1)]
                )
            except ValueError:
                vars.extend([prefix + chr(x) for x in range(ord(start), ord(end) + 1)])
    return vars


def reverse_func(start, end):
    def f(x):
        return start + end - x

    return f


def subtract_func(y=1):
    def f(x):
        return x - y

    return f


def recode_func(y=2):
    def f(x):
        return 0 if x == y else x

    return f


def combine_funcs(funcs):
    def f(x):
        for subf in funcs:
            x = subf(x)
        return x

    return f


def range_func(range_start, range_end):
    if range_start == -1 and range_end == -1:

        def always_true(x):
            return True

        return always_true

    def inrange(x):
        return isinstance(x, list) or (x >= range_start and x <= range_end)

    return inrange


def apply_func(range_start, range_end, recode_NA, reverse, subtract, recode_2to0, op):
    inrange = range_func(range_start, range_end)
    change_func = combine_funcs(
        [
            func
            for func, is_enabled in [
                (reverse_func(range_start, range_end), reverse),
                (subtract_func(1), subtract),
                (recode_func(), recode_2to0),
            ]
            if is_enabled
        ]
    )

    def f(x):
        changed_values = []
        for index, item in zip(x.index, x):
            if "_intermediate" in index:
                changed_values.append(item)
            else:
                changed_values.append(
                    np.nan if not inrange(item) else change_func(item)
                )
        scale_val = op(pd.Series(changed_values, index=x.index))
        return scale_val

    return f


class PositionalSeries(pd.Series):
    """Integer keys on a labelled index are positions, as in pandas 1.4.

    The scale functions were written against pandas 1.4.3 (their pinned
    environment) and index rows with x[0]; pandas 2 and later treat that as a label.
    """

    @property
    def _constructor(self):
        return PositionalSeries

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.iloc[key]
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, (int, np.integer)):
            self.iloc[key] = value
        else:
            super().__setitem__(key, value)


def custom_scale(func_name, range_start, range_end, subdf, meta):
    inrange = range_func(range_start, range_end)

    def custom_apply_func(func_name):
        def f(x):
            changed_values = PositionalSeries(
                x.apply(lambda x: np.nan if not inrange(x) else x)
            )
            # Some custom functions take the Stata metadata as a second argument.
            if len(signature(custom_func_map[func_name]).parameters) > 1:
                return custom_func_map[func_name](changed_values, meta)
            return custom_func_map[func_name](changed_values)

        return f

    return subdf.apply(custom_apply_func(func_name), axis=1)


op_map = {
    "mean": lambda x: x.mean(skipna=False),
    "mean_ignoreNA": lambda x: x.mean(skipna=True),
    "sum": lambda x: x.sum(skipna=False),
    "moduloandsum": lambda x: ((x + 1) % 3).sum(skipna=False),
    "moduloandsum_2": lambda x: (((x + 1) % 2)).sum(skipna=False),
    "moduloandmean": lambda x: ((x + 1) % 3).mean(skipna=False),
    "identity": lambda x: x,
    "identity_list": lambda x: list(x),
    "wtmean_4": lambda x: (x.sum(skipna=False) / 4.0 if x.min() >= 0 else np.nan),
    "wtmean_5": lambda x: (x.sum(skipna=False) / 5.0 if x.min() >= 0 else np.nan),
    "wtmean_8": lambda x: (x.sum(skipna=False) / 8.0 if x.min() >= 0 else np.nan),
    "wtmean_10": lambda x: (x.sum(skipna=False) / 10.0 if x.min() >= 0 else np.nan),
    "wtmean_11": lambda x: (x.sum(skipna=False) / 11.0 if x.min() >= 0 else np.nan),
    "wtmean_13": lambda x: (x.sum(skipna=False) / 13.0 if x.min() >= 0 else np.nan),
    "wtmean_15": lambda x: (x.sum(skipna=False) / 15.0 if x.min() >= 0 else np.nan),
    "mean_80pctNA": lambda x: (x.mean() if x.count() / x.size >= 0.8 else np.nan),
}

custom_func_map = {
    "unit_standardization": unit_standardization,
    "unit_standardization_per": unit_standardization_per,
    "month_year": month_year,
    "CBCL_combo": CBCL_combo,
    "conflict_tactics": conflict_tactics,
    "list_mean": list_mean,
    "list_mean_80pct": list_mean_80pct,
    "food_insecurity": food_insecurity,
    "familyMentalHealth": familyMentalHealth,
    "reverseSubset_andsubOne_mean": reverseSubset_andsubOne_mean,
    "reverseSubset_andsubOne_sum": reverseSubset_andsubOne_sum,
    "HOME_homeInterior": Home_homeInterior,
    "sdq": sdq,
    "sdq_externalizing": sdq_externalizing,
    "sdq_internalizing": sdq_internalizing,
    "dep_liberal_m5": dep_liberal_m5,
    "dep_liberal_f5": dep_liberal_f5,
    "dep_liberal_n5": dep_liberal_n5,
    "pubertal_p5": pubertal_p5,
    "dep_liberal_m4": dep_liberal_m4,
    "dep_liberal_f4": dep_liberal_f4,
    "HOME_devStim4": HOME_devStim4,
    "HOME_accessToReading4": HOME_accessToReading4,
    "HOME_outingActivities4": HOME_outingActivities4,
    "dep_liberal_m3": dep_liberal_m3,
    "dep_liberal_f3": dep_liberal_f3,
    "alcohol_m3": alcohol_m3,
    "alcohol_f3": alcohol_f3,
    "drug_m3": drug_m3,
    "drug_f3": drug_f3,
    "HOME_devStim3": HOME_devStim3,
    "dep_liberal_m2": dep_liberal_m2,
    "dep_liberal_f2": dep_liberal_f2,
    "emotionality_m2": emotionality_m2,
    "emotionality_f2": emotionality_f2,
}


def recode(
    df,
    scale_name,
    range_str,
    original_values,
    op,
    recode_NA=False,
    reverse=False,
    subtract=False,
    recode_2to0=False,
    custom_func_name="",
    meta=None,
):
    vars = get_var_list(range_str)
    subdf = df[vars]
    subdf = subdf.apply(
        lambda x: x.apply(
            lambda x: float(x.split(" ")[0]) if isinstance(x, str) else x
        ),
        axis=1,
    )
    range_start, range_end = original_values.split(":")
    range_start = int(range_start)
    range_end = int(range_end)
    if custom_func_name != "":
        df[scale_name] = custom_scale(
            custom_func_name, range_start, range_end, subdf, meta
        )
    else:
        df[scale_name] = subdf.apply(
            apply_func(
                range_start, range_end, recode_NA, reverse, subtract, recode_2to0, op
            ),
            axis=1,
        )


class StataLabels:
    """The part of pyreadstat's metadata object that unit_standardization reads."""

    def __init__(self, variable_value_labels):
        self.variable_value_labels = variable_value_labels


def compute_scales(df, instructions, names, meta):
    """Add the scales in ``names`` and the intermediates they use to ``df``.

    ``instructions`` is the parsable instructions table, rows run in its order
    as in get_scale_data. Returns the computed instruction names.
    """
    instructions = instructions.fillna(value="")
    rows = {row.metadata_new_name: row for row in instructions.itertuples()}
    needed = set()

    def require(name):
        if name in needed:
            return
        needed.add(name)
        for var in get_var_list(rows[name].varstr):
            if var in rows and var != name:
                require(var)

    for name in names:
        require(name)
    for row in instructions.itertuples():
        if row.metadata_new_name not in needed:
            continue
        recode(
            df,
            row.metadata_new_name,
            row.varstr,
            row.original,
            recode_NA=row.recodeNA,
            reverse=row.reverse,
            subtract=row.subtract1,
            recode_2to0=row.recode2to0,
            op=op_map[row.op],
            custom_func_name=row.custom,
            meta=meta,
        )
    return needed
