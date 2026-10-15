"""Scale construction from Cantrell et al., predicting-life-outcomes-ffcws.

custom_scales.py, hetero_units.py and wave2.py to wave5.py are copied unchanged
from scales/ of https://github.com/emilycantrell/predicting-life-outcomes-ffcws
(commit 995c845, Apache License 2.0). recode.py is adapted; its header lists
the changes.
"""

from .recode import StataLabels, compute_scales, get_var_list

__all__ = ["StataLabels", "compute_scales", "get_var_list"]
