import numpy as np
from scipy.optimize import curve_fit

MeV2erg = 1.6022e-6
sec2day = 1./86400.
c_light = 2.99792458e10
msol = 1.989e33
Navo = 6.02214076e23
kb_ergK = 1.38064852e-16
amu_mev = 931.49410372
amu_g = amu_mev * 1.7826619e-27
s_cgs = amu_g / kb_ergK

tau_ni = 8.76/sec2day      # s, average lifetime = half-life/ln2 = 6.075/ln2
q_ni = 2.133*MeV2erg  # Q value for 56Ni to 56Co
tau_co = 111.5/sec2day     # s, average lifetime = half-life/ln2 = 77.27/ln2
q_co = 4.567*MeV2erg  # Q value for 56Co to 56Fe
tau_n = 881.487               # s, average lifetime = half-life/ln2 = 611/ln2
q_n = 0.782*MeV2erg    # Q value for neutron to proton


def slope(t, b):
    a = -1.3
    return a * np.log10(t) + b

def fit_slope(tfit, qfit):
  tfit = np.geomspace(0.1, 10, 100)
  popt, pcov = curve_fit(slope, tfit, np.log10(qfit))
  return slope(tfit, *popt)


def get_2017_lum():
    time, lum, err = np.loadtxt("L_AT2017gfo.dat", unpack=True)
    return time, lum, err

def plot_2017_lum(ax):
    time, lum, err = get_2017_lum()
    up_err = 10**(lum+err) - 10**lum
    low_err = 10**lum - 10**(lum-err)
    ax.errorbar(time, 10**lum, yerr=[low_err, up_err], fmt='.', color='black', label="Smartt et al. 2017")

def print_time(tt):
    if tt < 1e-4:
        return f'$t = {tt*1e3:.1e}$ ms'
    if tt < 1:
        return f'$t = {tt*1e3:.1f}$ ms'
    if tt < 60:
        return f'$t = {tt:.1f}$ s'
    if tt < 3600:
        return f'$t = {tt/60:.1f}$ min'
    if tt < 86400:
        return f'{tt/3600:.1f} h'
    return f'$t = {tt/86400:.1f}$ d'
