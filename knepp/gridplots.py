from typing import Callable, Iterable
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.colorbar import Colorbar

from .util import print_time, c_light
from .knecsim import KnecSim

class KnecFigure:
    plots: list["Plot"]

    def __init__(
        self,
        fig: Figure,
        time: NDArray[np.float64],
        rad: NDArray[np.float64] = np.linspace(0, .8, 100),
        radial_coord: str = 'r-rin/t',
        display_time: bool = True,
        post_init: Callable[[], None] = lambda: None
    ):
        self.fig = fig
        self.plots = []
        self.time = time
        self.radial_coord = radial_coord
        self.rad = rad
        dr = rad[1] - rad[0]
        self.rad_edg = np.append(rad - dr/2, rad[-1] + dr/2)
        self.display_time = display_time
        self.post_init = post_init

    def add_animation(self, ani: "Plot"):
        self.plots.append(ani)

    def _init(self):
        ret = sum((ani.init(self.time) for ani in self.plots), [])
        self.post_init()
        return ret

    def _update(self, time: float) -> list:
        if self.display_time:
            self.fig.suptitle(print_time(time), x=.72, y=.88)
        return sum((ani.update(time) for ani in self.plots), [])

    def animate(self, **kwargs) -> FuncAnimation:
        return FuncAnimation(
            self.fig,
            self._update,
            self.time,
            init_func=self._init,
            **kwargs
        )

    def plot_data(self, time: float):
        self._init()
        self._update(time)


class Plot(ABC):

    def __init__(self, ax: Axes, fig: KnecFigure, **kwargs):
        self.ax = ax
        self.fig = fig
        self.fig.add_animation(self)
        self.kwargs = kwargs
        self.time = self.fig.time

    @abstractmethod
    def init(self, time: NDArray[np.float64]) -> list:
        ...

    @abstractmethod
    def update(self, time: float) -> list:
        ...


class SimPlot(Plot, ABC):
    def __init__(
        self,
        sim: KnecSim,
        t0: (float | None) = None,
        loc: str = 'top right',
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sim = sim
        self.t0 = t0

        self.loc = loc
        theta = sim.theta
        if sim.bitant:
            theta = theta[:int(np.ceil(len(theta)/2))].copy()
        if self.loc == 'top right':
            self.theta = 90 - theta
        elif self.loc == 'top left':
            self.theta = 90 + theta
        elif self.loc == 'bottom left':
            self.theta = 270 - theta
        elif self.loc == 'bottom right':
            self.theta = 270 + theta

        self.theta *= np.pi/180

        dth = self.theta[1] - self.theta[0]
        self.theta_edg = np.append(self.theta - dth/2, self.theta[-1] + dth/2)
        if sim.bitant:
            self.theta_edg[-1] -= dth/2

        self.rad = self.fig.rad
        self.rad_edg = self.fig.rad_edg
        self.radial_coord = self.fig.radial_coord

    def get_rad(self, time: (float | NDArray[np.float64])) -> NDArray[np.float64]:
        if self.radial_coord == "r-rin/t":
            r_ini = self.sim.get_init_cond('rad_initial', bitant=False)
            rr = self.sim.get_at_time(time, "radius", bitant=False)
            if isinstance(time, np.ndarray):
                time = time[:, np.newaxis, np.newaxis]
            return (rr-r_ini)/time/c_light
        elif self.radial_coord == "r/t":
            rr = self.sim.get_at_time(time, "radius", bitant=False)
            return rr/(time[..., None, None])/c_light

        elif self.radial_coord == "r/t+tsim":
            rr = self.sim.get_at_time(time, "radius", bitant=False)
            if self.sim.tsim is None:
                raise ValueError(f"simulation {self.sim.name} does not have tsim")
            return rr/(time[..., None, None]+self.sim.tsim)/c_light
        elif self.radial_coord == "r":
            return self.sim.get_at_time(time, "radius", bitant=False)
        elif self.radial_coord == "rkm":
            return self.sim.get_at_time(time, "radius", bitant=False)/1e5
        elif self.radial_coord == "rmkm":
            return self.sim.get_at_time(time, "radius", bitant=False)/1e14
        elif self.radial_coord == "rlog":
            return np.log10(self.sim.get_at_time(time, "radius", bitant=False))
        elif self.radial_coord == "v":
            return self.sim.get_at_time(time, "vel", bitant=False)/c_light

        raise ValueError(f"radial mode {self.radial_coord} not implemented")


class ColorPlot(SimPlot, ABC):
    im: QuadMesh
    cax: Axes
    cbar: Colorbar
    data: NDArray[np.float64]

    def __init__(
        self,
        cbar: (bool | Axes) = True,
        scale_norm: (Callable | None) = None,
        label: (str | None) = None,
        func: (Callable | None) = None,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if isinstance(cbar, Axes):
            self.cax = cbar
            self.draw_cbar = True
        elif cbar:
            self._create_cax()
            self.draw_cbar = True
        else:
            self.draw_cbar = False

        self.scale_norm = scale_norm
        self.label = label
        self.func = func

    @abstractmethod
    def load_data(self, time: NDArray[np.float64]):
        ...

    @abstractmethod
    def get_data(self, time: float) -> NDArray[np.float64]:
        ...

    def init(self, time) -> list:

        if self.t0 is None:
            self.t0 = time[0]

        self.rads = self.get_rad(time)
        self.load_data(time)

        data = self.get_data(self.t0)
        if self.func is not None:
            data = self.func(data, time=self.t0, plot=self)

        self.im = self.ax.pcolormesh(self.theta_edg, self.rad_edg, data, **self.kwargs)
        self.vmin = self.im.norm.vmin
        self.vmax = self.im.norm.vmax

        artists: list = [self.im]

        if self.draw_cbar:
            self._draw_cbar()
            artists.append(self.cax)
        self._set_label()
        return artists

    def update(self, time: float) -> list:
        if self.scale_norm is not None:
            vmin, vmax = self.scale_norm(time/self.t0, (self.vmin, self.vmax))
            norm = type(self.im.norm)(vmin=vmin, vmax=vmax)
            self.im.set_norm(norm)

        data = self.get_data(time)
        if self.func is not None:
            data = self.func(data, time=time, plot=self)

        self.im.set_array(data)

        if self.draw_cbar:
            self.cbar.update_normal(self.im)
            # self.cax.set_ylim(self.im.norm.vmin, self.im.norm.vmax)
            if 'left' in self.loc:
                self.cax.yaxis.set_label_position('left')
                self.cax.yaxis.set_ticks_position('left')
            return [self.ax, self.cax]

        return [self.ax]

    def _draw_cbar(self):

        self.cbar = plt.colorbar(self.im, cax=self.cax)

    def _set_label(self):
        if self.label is None:
            return

        if self.draw_cbar:
            self.cax.set_ylabel(self.label, )
            if  'left' in self.loc:
                self.cax.yaxis.set_label_position('left')
                self.cax.yaxis.set_ticks_position('left')
            return

        if self.loc == 'top right':
            self.ax.text(0.95, 0.95, self.label, transform=self.ax.transAxes, ha='right')
        elif self.loc == 'top left':
            self.ax.text(0.05, 0.95, self.label, transform=self.ax.transAxes, ha='left')
        elif self.loc == 'bottom left':
            self.ax.text(0.05, 0.05, self.label, transform=self.ax.transAxes, ha='left')
        elif self.loc == 'bottom right':
            self.ax.text(0.95, 0.05, self.label, transform=self.ax.transAxes, ha='right')

    def _create_cax(self):
        if self.loc=='top right':
            self.cax = self.ax.figure.add_axes((0.8, 0.52, 0.03, 0.36))
        elif self.loc=='top left':
            self.cax = self.ax.figure.add_axes((0.19, 0.52, 0.03, 0.36))
        elif self.loc=='bottom left':
            self.cax = self.ax.figure.add_axes((0.19, 0.11, 0.03, 0.36))
        elif self.loc=='bottom right':
            self.cax = self.ax.figure.add_axes((0.8, 0.11, 0.03, 0.36))

        if  'left' in self.loc:
            self.cax.yaxis.set_label_position('left')
            self.cax.yaxis.set_ticks_position('left')


class IsoContourPlot(SimPlot, ABC):

    def __init__(
        self,
        file_name: str,
        scale_levels: (Callable | None) = None,
        func: (Callable | None) = None,
        *args, **kwargs
    ):
        self.file_name = file_name
        self.scale_levels = scale_levels
        self.func = func
        super().__init__(*args, **kwargs)

    @abstractmethod
    def load_data(self, time: NDArray[np.float64]):
        ...

    @abstractmethod
    def get_data(self, time: float) -> NDArray[np.float64]:
        ...

    def init(self, time) -> list:

        if self.t0 is None:
            self.t0 = time[0]

        self.rads = self.get_rad(time)
        self.load_data(time)

        data = self.get_data(self.t0)
        if self.func is not None:
            data = self.func(data, time=self.t0, plot=self)

        self.im = self.ax.contour(self.theta, self.rad, data, **self.kwargs)

        self.levels = self.im.levels

        return [self.im]

    def update(self, time: float) -> list:
        self.im.remove()

        if self.scale_levels is not None:
            self.kwargs['levels'] = self.scale_levels(time, self.levels)

        data = self.get_data(time)
        if self.func is not None:
            data = self.func(data, time=time, plot=self)

        self.im = self.ax.contour(self.theta, self.rad, data, **self.kwargs)

        return [self.im]

class InitCondPlot(ColorPlot):

    def __init__(self, file_name: str, *args, **kwargs,):
        self.file_name = file_name
        super().__init__(*args, **kwargs)

    def load_data(self, time: NDArray[np.float64]):
        self.data = self.sim.get_init_cond(self.file_name, bitant=False)

    def get_data(self, time) -> NDArray[np.float64]:
        it = self.time.searchsorted(time)
        interp_data = np.zeros((len(self.rad), len(self.theta)))*np.nan
        for ith, _ in enumerate(self.theta):
            interp_data[:, ith] = interp1d(self.rads[it, :, ith], self.data[:, ith],
                                           bounds_error=False,
                                           fill_value=np.nan)(self.rad)
        return interp_data


class XGPlot(ColorPlot):

    def __init__(self, file_name: str, *args, **kwargs,):
        self.file_name = file_name
        super().__init__(*args, **kwargs)

    def load_data(self, time):
        self.data = self.sim.get_at_time(time, self.file_name, bitant=False)

    def get_data(self, time: float) -> NDArray[np.float64]:
        it = self.time.searchsorted(time)
        interp_data = np.zeros((len(self.rad), len(self.theta)))*np.nan
        for ith, _ in enumerate(self.theta):
            interp_data[:, ith] = interp1d(self.rads[it, :, ith], self.data[it, :, ith],
                                           bounds_error=False,
                                           fill_value=np.nan)(self.rad)
        return interp_data

class AbundancePlot(ColorPlot):
    def __init__(self, A, Z, *args, **kwargs):
        self.A = A
        self.Z = Z
        super().__init__(*args, **kwargs)

    def load_data(self, time: NDArray[np.float64]):
        self.data = self.sim.get_isotope_at_time(time, self.A, self.Z, bitant=False)

    get_data = XGPlot.get_data

class XGIsoContourPlot(IsoContourPlot):
    load_data = XGPlot.load_data
    get_data = XGPlot.get_data

class InitCondIsoContourPlot(IsoContourPlot):
    load_data = InitCondPlot.load_data
    get_data = InitCondPlot.get_data

class PhotospherePlot(SimPlot):
    def init(self, time: NDArray[np.float64]) -> list:
        i_ph = self.sim.get_at_time(time, "index_photo", bitant=False)
        i_ph = i_ph.astype(int) - 1
        i_ph[i_ph==0] = 1 # sometimes the last shell can be weirdly slow
        if self.fig.radial_coord.startswith('r'):
            rad = self.sim.get_at_time(time, 'radius', bitant=False)
            self.r_photo = np.zeros((len(time), i_ph.shape[-1]))
            for th, _ in enumerate(self.theta):
                self.r_photo[:, th] = rad[np.arange(rad.shape[0]), i_ph[:, th], th]
        if self.fig.radial_coord == "r-rin/t":
            r_ini = self.sim.get_init_cond('rad_initial', bitant=False)
            for th, _ in enumerate(self.theta):
                self.r_photo[:, th] = self.r_photo[:, th] - r_ini[i_ph[:,th], th]
            self.r_photo = self.r_photo / time[:, np.newaxis] / c_light
        elif self.fig.radial_coord == "r/t":
            self.r_photo /= (time[:, np.newaxis]) * c_light
        elif self.fig.radial_coord == "r/t+tsim":
            self.r_photo /= (time[:, np.newaxis] + self.sim.tsim) * c_light
        elif self.fig.radial_coord == "rkm":
            self.r_photo /= 1e5
        elif self.fig.radial_coord == "rmkm":
            self.r_photo /= 1e14
        elif self.fig.radial_coord == "rlog":
            self.r_photo = np.log10(self.r_photo)
        elif self.fig.radial_coord == "v":
            self.r_photo = self.sim.get_at_time(time, 'vel', bitant=False)[i_ph]
            self.r_photo /= c_light

        self.li, = self.ax.plot(self.theta, self.r_photo[0], **self.kwargs)
        return [self.li]

    def update(self, time: float) -> list:
        it = self.time.searchsorted(time)
        self.li.set_ydata(self.r_photo[it])
        self.ax.set_ylim(0, self.fig.rad.max())
        return [self.li]



################################################################################

def powerlaw(exponent: float) -> Callable:
    def scale(time, levels:tuple[float, ...]) -> tuple[float, ...]:
        return tuple([level * time**exponent for level in levels])
    return scale


def normalize_by_section(data: NDArray[np.float64], **_):
    for ith in range(data.shape[1]):
        norm = data[:, ith][np.isfinite(data[:, ith])].max()
        data[:, ith] = 1 - data[:, ith] / norm
    return data

################################################################################
