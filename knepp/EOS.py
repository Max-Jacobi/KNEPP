from scipy.optimize import brentq
import numpy as np

from .utils import s_cgs

try:
    from helmeos import HelmTable

    heos = HelmTable().eos_DT
    helm_found = True
except ImportError:
    helm_found = False



def _temp_from_entr_single(
    rho,
    entr,
    ye,
    abar,
):
    assert helm_found, "helmeos not found"
    entr = entr / s_cgs

    def _inner(temp):
        res = heos(rho, temp, abar, ye * abar)["stot"]
        return res - entr

    return brentq(_inner, 1, 3e10)


def temp_from_entr(
    rho,
    entr,
    ye,
    abar,
):
    if isinstance(rho, (int, float)):
        return _temp_from_entr_single(rho, entr, ye, abar)
    return np.array(
        [_temp_from_entr_single(r, e, y, a)
         for r, e, y, a in zip(rho, entr, ye, abar)]
    )


def _rho_from_entr_single(
    temp,
    entr,
    ye,
    abar,
):
    assert helm_found, "helmeos not found"
    entr = entr / s_cgs

    def _inner(rho):
        res = heos(rho, temp, abar, ye * abar)["stot"]
        return res - entr

    return brentq(_inner, 1e-10, 1e12)


def rho_from_entr(
    temp,
    entr,
    ye,
    abar,
):
    if isinstance(temp, (int, float)):
        return _rho_from_entr_single(temp, entr, ye, abar)
    return np.array(
        [_rho_from_entr_single(t, e, y, a)
         for t, e, y, a in zip(temp, entr, ye, abar)]
    )
