import json
from functools import lru_cache
import numpy as np
from scipy.interpolate import interp1d
from numpy.typing import NDArray

from .knecrun import KnecRun
from .util import tqdm


class KnecSim:
    color: str | None = None
    verbose: bool
    tsim: float | None

    def __init__(
        self,
        path: str,
        bitant: bool = True,
        verbose: bool = False,
        dpath: str = "data",
    ):
        self.path = path
        self.name = path.split("/")[-1]
        self.bitant = bitant
        self.verbose = verbose
        self._read_metadata()
        self.sections = []
        self.runs = []
        for md in self.metadata:
            name = md["name"].replace("=", "")
            path = f"{self.path}/{name}"
            for section in self.sections:
                if section.path == path:
                    run = section.run
                    break
            else:
                run = KnecRun(
                    path, bitant=self.bitant, verbose=self.verbose, dpath=dpath
                )
                self.runs.append(run)
            self.sections.append(KnecSection(md, run, self))

        self.theta = np.array([sec.theta for sec in self.sections]) / np.pi * 180

        self.nsim = len(self.sections)

        for run in self.sections:
            run.md["geom_weight"] = 1 / run.scaling
            run.md["mass"] = run.mtot / run.scaling

        self.total_mass = sum(run.mass for run in self.sections)

        for run in self.sections:
            run.md["mass_weight"] = run.mass / self.total_mass

        self.glob_time = np.geomspace(
            1e-5, min(run.tmax for run in self.sections), 1000
        )

        try:
            ej_time = self.get_init_cond("ejection_time", bitant=False)
            self.tsim = ej_time.max()
        except FileNotFoundError:
            self.tsim = None

    def _read_metadata(self):
        with open(self.path + "/metadata.json") as f:
            self.metadata = json.load(f)

    def flux_factors(self, angle):
        ang, *ffs = np.loadtxt("flux_factors_51.dat", unpack=True)
        ffs = np.array(ffs) / np.pi
        return interp1d(ang, ffs, kind="cubic")(angle)

    def _combine(self, weights, func, **kwargs):
        q_glob = np.zeros_like(self.glob_time)
        for ww, run in tqdm(
            zip(weights, self.sections),
            desc=f"Combining {func.__name__} for {self.name}",
            total=self.nsim,
            disable=not self.verbose,
            ncols=0,
            leave=False,
            unit="Sections",
        ):
            t_shell, q_shell = func(run, **kwargs)
            q_loc = ww * np.interp(self.glob_time, t_shell, q_shell)
            q_loc[np.isnan(q_loc)] = 0
            q_glob += q_loc
        return self.glob_time, q_glob

    @lru_cache(maxsize=None)
    def mass_combine(self, func, **kwargs):
        return self._combine(
            [run.mass_weight for run in self.sections],
            func,
            **kwargs,
        )

    @lru_cache(maxsize=None)
    def geom_combine(self, func, **kwargs):
        return self._combine([run.geom_weight for run in self.sections], func, **kwargs)

    @lru_cache(maxsize=None)
    def flux_factor_combine(self, func, angle, **kwargs):
        return self._combine(self.flux_factors(angle), func, **kwargs)

    def get_at_time(
        self,
        time: float | NDArray[np.float64],
        file_name: str,
        bitant: bool = True,
    ) -> NDArray[np.float64]:
        if bitant:
            data = np.array(
                [sec.load_at_time(time, file_name) for sec in self.sections]
            )
        else:
            data = np.array([sec.load_at_time(time, file_name) for sec in self.runs])
        data = data.T  # swap theta and mass coord
        if len(data.shape) == 3:
            data = data.swapaxes(0, 1)  # bring time to the front
        return data

    def get_init_cond(self, file_name: str, bitant: bool = True) -> NDArray[np.float64]:
        if bitant:
            data = np.array(
                [
                    sec.load_dat(f"initial_conditions/{file_name}")[1]
                    for sec in self.sections
                ]
            )
        else:
            data = np.array(
                [
                    sec.load_dat(f"initial_conditions/{file_name}")[1]
                    for sec in self.runs
                ]
            )
        return data.T

    def get_isotope_at_time(
        self,
        time: float | NDArray[np.float64],
        A: int,
        Z: int,
        bitant: bool = True,
    ) -> NDArray[np.float64]:
        if bitant:
            data = np.array(
                [sec.load_isotope_at_time(time, A, Z) for sec in self.sections]
            )
        else:
            data = np.array([sec.load_isotope_at_time(time, A, Z) for sec in self.runs])
        data = data.T  # swap theta and mass coord
        if len(data.shape) == 3:
            data = data.swapaxes(0, 1)  # bring time to the front
        return data


class KnecSection:
    md: dict
    run: KnecRun

    def __init__(self, md: dict, run: KnecRun, sim: KnecSim):
        md["sim"] = sim
        self.md = md
        self.run = run

    def __str__(self):
        return f"KnecSection at theta={self.theta} ({self.run})"

    def __repr__(self):
        return f"KnecSection({self.run})"

    def __getattr__(self, attr):
        if attr in self.md:
            return self.md[attr]
        return getattr(self.run, attr)
