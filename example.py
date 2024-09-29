################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import knepp as kp

################################################################################

sim = kp.KnecSim("path/to/simulation", verbose=True)

radial_coord = "r/t"
rad = np.linspace(0, 1, 200)

################################################################################


def make_plot(kfig, ax, sim):
    # common keyword arguments for all plots
    kw = dict(ax=ax, fig=kfig, sim=sim)

    # initial Ye
    kp.InitCondPlot(
        "initial_ye",
        cmap="seismic_r",
        vmin=0,
        vmax=0.6,
        loc="top left",
        label="$Y_e$",
        **kw,
    )

    # Ni56 abundance
    # This might take very long to load
    # The abundances are saved in a separate dataset so future loads will be fast
    kp.AbundancePlot(
        A=56,
        Z=28,
        cmap="nipy_spectral",
        norm=LogNorm(1e-5, 1, clip=True),
        loc="bottom left",
        label=r"$X({\rm ^{56}Ni})$",
        **kw,
    )

    # fgamma
    kp.XGPlot(
        "fgamma",
        cmap="hot",
        loc="top right",
        label=r"$f_\gamma$",
        norm=LogNorm(1e-4, 1, clip=True),
        **kw,
    )

    # heating rate

    # define custom Plot object based od XGPlot plot rho * heating_rate
    class HeatPlot(kp.XGPlot):
        def __init__(self, *args, **kwargs):
            # normally init requires file_name but we set it in load_data
            super().__init__(file_name="", *args, **kwargs)

        def load_data(self, time):
            self.file_name = "rho"
            super().load_data(time)
            rho = self.data
            self.file_name = "radioactive_power"
            super().load_data(time)
            self.data = self.data * rho

    HeatPlot(
        cmap="terrain",
        loc="bottom right",
        norm=LogNorm(1e-2, 1e-9),
        label=r"$\dot{q}$ [erg cm$^{-3}$ s$^{-1}$]",
        **kw,
    )

    levels = (1e-18, 1e-16)
    for loc in ("top left", "top right", "bottom right", "bottom left"):
        kp.PhotospherePlot(color="fuchsia", loc=loc, **kw)
        kp.XGIsoContourPlot("rho", levels=levels, colors="w", linestyles='--', loc=loc, **kw)


################################################################################

fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

ax.set_xlabel(r"$r/t$ [$c$]")
ax.set_title(sim.name)
ax.grid(alpha=0.5)

tplot = 86400 * 2
# base level object that encapsulates subplots and manages plotting/animations
kfig = kp.KnecFigure(fig, time=np.array([tplot]), rad=rad, radial_coord=radial_coord)

make_plot(kfig, ax, sim)

# make a single plot
kfig.plot_data(time=tplot)

plt.show()

################################################################################

fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

ax.set_xlabel(r"$r/t$ [$c$]")
ax.set_title(sim.name)
ax.grid(alpha=0.5)

# time grid for animation
time = np.geomspace(1e-1, 5, 100) * 86400

kfig = kp.KnecFigure(fig, time=time, rad=rad, radial_coord=radial_coord)

make_plot(kfig, ax, sim)

# get FuncAnimation object
ani = kfig.animate()

# save as mp4
ani.save("example.mp4", fps=10)
