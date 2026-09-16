"""Local sklearn linear estimators with conditional LAPACK recovery.

Keep sklearn's preprocessing, constraints, rank cutoffs and CV implementation.
Only the numerical calls in private copies of its functions are replaced; no
process-wide monkey patches (workers and parallel CV may fit concurrently).
These bindings are tested against the scikit-learn version pinned by NK_Grid.
"""
from __future__ import annotations

import inspect
from types import FunctionType

from sklearn.base import _fit_context
from sklearn.linear_model import LinearRegression as _LinearRegression
from sklearn.linear_model import Ridge as _Ridge
from sklearn.linear_model import RidgeCV as _RidgeCV
from sklearn.linear_model import _ridge

from .svd_fallback import safe_lstsq, safe_svd


class _Proxy:
    def __init__(self, original, **overrides):
        self.original = original
        self.overrides = overrides

    def __getattr__(self, name):
        return self.overrides.get(name, getattr(self.original, name))


def _bind(function, **replacements):
    """Copy a function's globals without changing the installed sklearn module."""
    original = inspect.unwrap(function)
    missing = replacements.keys() - original.__globals__.keys()
    if missing:
        raise RuntimeError(f"Unsupported sklearn numerical bindings: {missing}")
    namespace = dict(original.__globals__, **replacements)
    bound = FunctionType(original.__code__, namespace, original.__name__,
                         original.__defaults__, original.__closure__)
    bound.__kwdefaults__ = original.__kwdefaults__
    return bound


def _namespace_binding(function, name):
    original = inspect.unwrap(function).__globals__[name]

    def get_namespace(*args, **kwargs):
        xp, *rest = original(*args, **kwargs)
        return (_Proxy(xp, linalg=_Proxy(xp.linalg, svd=lambda a, **kw:
                safe_svd(a, primary=xp.linalg.svd, **kw))), *rest)

    return get_namespace


_linear_fit = inspect.unwrap(_LinearRegression.fit)


class LinearRegression(_LinearRegression):
    fit = _fit_context(prefer_skip_nested_validation=True)(_bind(
        _linear_fit, linalg=_Proxy(_linear_fit.__globals__["linalg"], lstsq=safe_lstsq)))


_solve_svd = _bind(_ridge._solve_svd, get_namespace=_namespace_binding(
    _ridge._solve_svd, "get_namespace"))
_ridge_regression = _bind(_ridge._ridge_regression, _solve_svd=_solve_svd)


class _RecoveringBaseRidge(_ridge._BaseRidge):
    fit = _bind(_ridge._BaseRidge.fit, _ridge_regression=_ridge_regression)


class Ridge(_Ridge, _RecoveringBaseRidge):
    # Original Ridge.fit validates inputs, then super() resolves to our base.
    pass


class _RecoveringRidgeGCV(_ridge._RidgeGCV):
    _svd_decompose_design_matrix = _bind(
        _ridge._RidgeGCV._svd_decompose_design_matrix,
        get_namespace_and_device=_namespace_binding(
            _ridge._RidgeGCV._svd_decompose_design_matrix, "get_namespace_and_device"))


class RidgeCV(_RidgeCV):
    fit = _fit_context(prefer_skip_nested_validation=True)(_bind(
        _ridge._BaseRidgeCV.fit, _RidgeGCV=_RecoveringRidgeGCV, Ridge=Ridge))
