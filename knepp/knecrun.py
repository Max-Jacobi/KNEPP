from functools import lru_cache
import os
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from h5py import File

from .composition import Composition
from .util import Navo, tqdm


class KnecRun:

    color: Optional[str] = None
    nice_name: str
    path: str
    dpath: str
    bitant: bool
    dm: NDArray[np.float64]
    mtot: float
    weights: NDArray[np.float64]
    imax: int
    verbose: bool

    def __init__(
        self, path: str, dpath: str = "data", bitant: bool = True, verbose: bool = False
    ):
        self.path = path
        self.dpath = os.path.join(path, dpath)
        self.name = os.path.basename(path)
        self.nice_name = self.name.replace("_", " ")
        self.bitant = bitant
        self.verbose = verbose
        for xg in "radioactive_power mass radius".split():
            try:
                self.tmax = self.load_xg(xg)[0].max()
                break
            except FileNotFoundError:
                continue
        else:
            raise FileNotFoundError(f"Could not find any xg or h5 files for initialization")

        self._comp = None

        self.dm = np.loadtxt(
            f"{self.dpath}/initial_conditions/delta_mass_initial.dat",
            unpack=True,
            usecols=1,
        )
        self.mtot = self.dm.sum()
        self.weights = self.dm / self.mtot
        self.imax = len(self.dm)

    @property
    def comp(self) -> Composition:
        if self._comp is not None:
            return self._comp
        try:
            self._comp = Composition(f"{self.dpath}/comp.h5")
            self._comp.verbose = self.verbose
            return self._comp
        except OSError as er:
            raise OSError(f"No composition data available for {self}") from er

    def __repr__(self) -> str:
        return f"KnecRun({self.path})"

    def __str__(self) -> str:
        return self.name

    @lru_cache(maxsize=26)
    def load_xg(
        self, file_name: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

        if os.path.isfile(f"{self.dpath}/{file_name}.h5"):
            return self.load_h5(file_name)

        with open(f"{self.dpath}/{file_name}.xg", "r") as f:
            times = []
            data = []
            lines = f.readlines()
            for line in tqdm(
                lines,
                ncols=0,
                total=len(lines),
                desc=f"Loading {self.dpath}/{file_name}.xg",
                leave=False,
            ):
                # skip empty lines
                if not line.strip():
                    continue
                if "Time" in line:
                    times.append(float(line.split("=")[1]))
                    data.append([])
                    dat_shell = data[-1]
                    continue
                mass, val = map(float, line.split())
                dat_shell.append([mass, val])

        data = np.array(data)[..., 1]
        times = np.array(times)
        return times, data

    def load_dat(
        self, file_name: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        t, d = np.loadtxt(f"{self.dpath}/{file_name}.dat", usecols=(0, 1), unpack=True)
        return t, d

    def load_h5(self, file_name) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        with File(f"{self.dpath}/{file_name}.h5", "r") as f:
            q = np.array(f["data"][:])
            t = np.array(f["times"][:])
        return t, q

    def load_at_time(
        self, time: float | NDArray[np.float64], file_name: str
    ) -> NDArray[np.float64]:
        if os.path.isfile(f"{self.dpath}/{file_name}.h5"):
            return interp1d(*self.load_h5(file_name), axis=0)(time)
        if os.path.isfile(f"{self.dpath}/{file_name}.xg"):
            return interp1d(*self.load_xg(file_name), axis=0)(time)
        if os.path.isfile(f"{self.dpath}/{file_name}.dat"):
            return interp1d(*self.load_dat(file_name), axis=0)(time)
        raise FileNotFoundError(f"Could not find {self.dpath}/{file_name}.[h5|xg|dat]")

    def load_isotope_at_time(
        self,
        time: float | NDArray[np.float64],
        A: int,
        Z: int,
    ) -> NDArray[np.float64]:
        if self.comp is None:
            raise ValueError(f"No composition data available for {self}")
        return interp1d(self.comp.times, self.comp.spec_abundance(A, Z), axis=0)(time)

    def mass_average(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        assert (
            data.shape[1] == self.imax
        ), f"Data shape={data.shape} does not match imax={self.imax} for {self}"
        return np.sum(data * self.weights[None, :], axis=1)

    def load_mass_avg(
        self, file_name: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        t, data = self.load_xg(file_name)
        return t, self.mass_average(data)

    def get_rest_heating_loc(self):
        time, qtot = self.load_xg("radioactive_power")
        qrest = qtot - sum(self.load_xg(f"radioactive_power_{sp}")[1]
                  for sp in "alphas els_pos gammas nus".split())
        return time, qrest

    def get_rest_heating(self):
        time, qq = self.get_rest_heating_loc()
        return time, self.mass_average(qq)


    def get_rest_therm_heating_loc(self, heating_fac=0.8):
        time, qq = self.get_rest_heating_loc()
        return time, qq*heating_fac

    def get_rest_therm_heating(self, heating_fac=0.8):
        time, qt = self.get_rest_therm_heating_loc(heating_fac)
        return time, self.mass_average(qt)

    def get_consv_therm_heating_loc(self):
        time = self.load_xg("radioactive_power_alphas")[0]
        qs = [self.load_xg(f"radioactive_power_{sp}")[1]
              for sp in "alphas els_pos gammas".split()]
        fs = [self.load_xg(f"f{sp}")[1] for sp in "alphas electrons gamma".split()]

        qq = sum(q*f for q,f in zip(qs, fs))

        return time, qq

    def get_consv_therm_heating(self):
        time, qt = self.get_consv_therm_heating_loc();
        return time, self.mass_average(qt)

    def get_consv_lum(self):
        tq, qt = self.get_consv_therm_heating_loc()
        tph, ph_idx = self.load_dat("index_photo")
        ph_idx = np.interp(tq, tph, ph_idx).round().astype(int) - 1

        if len(qt) == len(ph_idx)-1:
            ph_idx = ph_idx[:-1]

        for it, iph in enumerate(ph_idx):
            qt[it, :iph] = 0
        return tq, np.sum(qt * self.dm[None, :], axis=1)

    def get_AZ_heating_loc(
        self,
        A: int,
        Z: int,
        qovertau: float,
    ):
        y_full = self.comp.spec_abundance(A, Z)
        return self.comp.times, Navo * y_full * qovertau

    def get_AZ_heating(
        self,
        A: int,
        Z: int,
        qovertau: float,
    ):
        tt, qq = self.get_AZ_heating_loc(A, Z, qovertau)
        return tt, self.mass_average(qq)

    def get_AZ_therm_heating_loc(
        self,
        A: int,
        Z: int,
        qovertau: float,
        therm: tuple(tuple[float, str]) = (("thermalization_factor", 1.0),),
    ):
        qf = self.get_AZ_heating_loc(A, Z, qovertau)[1]

        ft = 0
        perc_tot = 0
        for filename, perc in therm:
            perc_tot += perc
            if filename == "neutrinos":
                continue
            ft += perc * self.load_at_time(self.comp.times, filename)
        assert perc_tot <= 1.0
        ft += (1 - perc_tot) * 0.5
        return self.comp.times, qf * ft

    def get_AZ_therm_heating(
        self,
        A: int,
        Z: int,
        qovertau: float,
        therm: tuple(tuple[float, str]) = (("thermalization_factor", 1.0),),
    ):
        tt, qq = self.get_AZ_therm_heating_loc(A, Z, qovertau, therm)
        return tt, self.mass_average(qq)

    def get_AZ_lum(
        self,
        A: int,
        Z: int,
        qovertau: float,
        therm: tuple(tuple[float, str]) = (("thermalization_factor", 1.0),),
    ):
        qt = self.get_AZ_therm_heating_loc(A, Z, qovertau, therm)[1]
        tph, ph_idx = self.load_dat("index_photo")
        ph_idx = np.interp(self.comp.times, tph, ph_idx).round().astype(int) - 1

        for it, iph in enumerate(ph_idx):
            qt[it, :iph] = 0
        return self.comp.times, np.sum(qt * self.dm[None, :], axis=1)

    def get_NiCo_heating(self):
        from util import q_ni, tau_ni, q_co, tau_co

        Qni = self.get_AZ_heating(56, 28, q_ni / tau_ni)[1]
        Qco = self.get_AZ_heating(56, 27, q_co / tau_co)[1]
        return self.comp.times, Qni + Qco

    def get_NiCo_therm_heating(self, **kwargs):
        from util import q_ni, tau_ni, q_co, tau_co

        Qni = self.get_AZ_therm_heating(56, 28, q_ni / tau_ni, **kwargs)[1]
        Qco = self.get_AZ_therm_heating(56, 27, q_co / tau_co, **kwargs)[1]
        return self.comp.times, Qni + Qco

    def get_NiCo_lum(self, **kwargs):
        from util import q_ni, tau_ni, q_co, tau_co

        Lni = self.get_AZ_lum(56, 28, q_ni / tau_ni, **kwargs)[1]
        Lco = self.get_AZ_lum(56, 27, q_co / tau_co, **kwargs)[1]
        return self.comp.times, Lni + Lco

    def get_neutron_heating(self):
        from util import q_n, tau_n

        qn = self.get_AZ_heating(1, 0, q_n / tau_n)[1]
        return self.comp.times, qn

    def get_neutron_therm_heating(self, **kwargs):
        from util import q_n, tau_n

        qn = self.get_AZ_therm_heating(1, 0, q_n / tau_n, **kwargs)[1]
        return self.comp.times, self.mass_average(qn)

    def get_neutron_lum(self, **kwargs):
        from util import q_n, tau_n

        L = self.get_AZ_lum(1, 0, q_n / tau_n, **kwargs)[1]
        return self.comp.times, L

    def get_photosphere_mass(self):
        tph, ph_idx = self.load_dat("index_photo")
        tph, ph_idx = tph[1:], ph_idx[1:].astype(int) - 1
        mass = np.zeros((len(tph), self.imax))
        for it, iph in enumerate(ph_idx):
            mass[it, :iph] = self.dm[:iph]
        return tph, mass.sum(axis=1)

    def export_trajectory(self, path, ish, prepend_homol_T=0.0):
        from .EOS import temp_from_entr, rho_from_entr

        tt, rho = self.load_xg("rho")
        ttemp, temp = self.load_xg("temp")
        tye, ye = self.load_xg("ye")
        trad, rad = self.load_xg("radius")
        assert np.all(tt == ttemp) and np.all(tt == tye) and np.all(tt == trad)

        if not os.path.isdir(path):
            os.mkdir(path)

        rh = rho[:, ish]
        te = temp[:, ish]
        yy = ye[:, ish]
        rr = rad[:, ish]

        if te[0] < prepend_homol_T:
            entr_e = self.load_xg("entropy")[1][0, ish]
            abar_e = self.load_xg("abar")[1][0, ish]
            vel_e = self.load_xg("vel")[1][0, ish]
            ye_e = yy[0]
            r_e = rr[0]
            t_e = tt[0]
            rho_e = rh[0]

            abar_i = np.ones_like(abar_e)

            rho_i = rho_from_entr(prepend_homol_T, entr_e, ye_e, abar_i)

            r_i = (rho_e / rho_i) ** (1 / 3) * r_e
            t_i = t_e + (r_i - r_e) / vel_e  # smaller than t_e

            t_pre = np.linspace(t_i, t_e, 100)
            r_pre = vel_e * (t_pre - t_e) + r_e
            rho_pre = rho_e * (r_pre / r_e) ** -3
            ye_pre = np.full_like(r_pre, ye_e)
            entr_pre = np.full_like(r_pre, entr_e)
            abar_pre = np.linspace(abar_i, abar_e, 100)
            temp_pre = temp_from_entr(rho_pre, entr_pre, ye_pre, abar_pre)

            if temp_pre[0] - prepend_homol_T > 1e6:
                print(f"Warning: in {self}: T_i = {temp_pre[0]*1e-9:.2e}GK")

            if temp_pre[-1] - te[0] > 1e6:
                print(
                    f"Warning: in {self}: T_e = {temp_pre[-1]*1e-9:.2e}GK, "
                    f"T_knec = {te[0]*1e-9:.2e}GK"
                )

            tt = np.concatenate((t_pre[:-1], tt))
            rh = np.concatenate((rho_pre[:-1], rh))
            te = np.concatenate((temp_pre[:-1], te))
            yy = np.concatenate((ye_pre[:-1], yy))
            rr = np.concatenate((r_pre[:-1], rr))

            tt -= tt[0]

        np.savetxt(
            f"{path}/trajectory.dat",
            np.column_stack((tt, rr, rh, te, yy)),
            fmt="%15.8e",
            header="time:s rad:cm dens temp:K ye",
        )

    def get_finab(self):
        y_f = self.comp.AZ[-1]
        y_f *= self.dm[:, None, None]
        return y_f.sum(axis=0) / self.dm.sum()

    def get_finabsum(self):
        return self.get_finab().sum(axis=1)

    def get_finabelem(self):
        return self.get_finab().sum(axis=0)
