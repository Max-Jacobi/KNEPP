from itertools import starmap
from scipy.optimize import brentq
import numpy as np

from .util import s_cgs

try:
    from helmeos import HelmTable

    heos = HelmTable().eos_DT
    helm_found = True
except ImportError:
    helm_found = False

_temp_bounds = 1, 3e10
_rho_bounds = 1e-10, 1e12

@np.vectorize
def temp_from_entr(rho, entr, ye, abar):
    assert helm_found, "helmeos not found"
    entr = entr / s_cgs
    def _inner(temp):
        return heos(rho, temp, abar, ye * abar)["stot"] - entr
    return brentq(_inner, *_temp_bounds)

@np.vectorize
def rho_from_entr(temp, entr, ye, abar):
    assert helm_found, "helmeos not found"
    entr = entr / s_cgs
    def _inner(rho):
        return heos(rho, temp, abar, ye * abar)["stot"] - entr
    return brentq(_inner, *_rho_bounds)

@np.vectorize
def temp_from_pres(rho, pres, ye, abar):
    assert helm_found, "helmeos not found"
    def _inner(temp):
        return heos(rho, temp, abar, ye * abar)["ptot"] - pres
    return brentq(_inner, *_temp_bounds)

@np.vectorize
def rho_from_pres(temp, pres, ye, abar):
    assert helm_found, "helmeos not found"
    def _inner(rho):
        return heos(rho, temp, abar, ye * abar)["ptot"] - pres
    return brentq(_inner, *_rho_bounds)

