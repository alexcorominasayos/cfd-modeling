"""3D plots of the result matrices."""

import matplotlib.pyplot as plt
import numpy as np

MAX_PLOT_POINTS = 100   # per axis: more points make the plot very slow and look the same


def plot_surface(times, positions, values, z_label, title):
    """Create a 3D surface figure of values[time, position].

    If there are more than MAX_PLOT_POINTS points in an axis, only some of them
    are drawn (we skip rows or columns evenly). The figure is not shown until
    show_plots() is called, so several figures can be opened together.
    """
    time_skip = max(1, len(times) // MAX_PLOT_POINTS)
    position_skip = max(1, len(positions) // MAX_PLOT_POINTS)

    time_grid, position_grid = np.meshgrid(
        times[::time_skip], positions[::position_skip], indexing="ij"
    )
    plotted_values = values[::time_skip, ::position_skip]

    figure = plt.figure(figsize=(10, 7))
    axes = figure.add_subplot(projection="3d")
    surface = axes.plot_surface(time_grid, position_grid, plotted_values, cmap="viridis")
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Position (m)")
    axes.set_zlabel(z_label)
    axes.set_title(title)
    figure.colorbar(surface, ax=axes, shrink=0.6, label=z_label)


def show_plots():
    """Open all the figures created with plot_surface."""
    plt.show()
