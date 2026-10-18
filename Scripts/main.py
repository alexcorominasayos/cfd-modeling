"""Velocity and temperature of a viscous fluid film flowing down an inclined plane.

Run it with:  python main.py
"""

import time

import numpy as np

import report
from config import FLUIDS, MAX_SAVED_STEPS
from physics import get_sigma, gravity_along_plane, momentum_diffusivity, thermal_diffusivity
from plotting import plot_surface, show_plots
from solvers import (build_grid, get_velocity_substeps, saved_steps, solve_temperature,
                     solve_velocity)
from user_input import ask_case

UNITS = {"velocity": "m/s", "temperature": "K"}


# --- Calculations ------------------------------------------------------------
def make_grid(case, quantity):
    """Return the positions, the saved times, n_steps and save_every of a quantity."""
    settings = case[quantity]
    n_cells, n_steps = build_grid(case["depth"], settings["time"], case["dy"], settings["dt"])
    positions = np.linspace(0, case["depth"], n_cells + 1)

    # Save only some time steps so very long simulations do not fill the memory.
    # The last step is always saved too, so the results end exactly at the final time.
    save_every = max(1, n_steps // MAX_SAVED_STEPS)
    times = saved_steps(n_steps, save_every) * settings["dt"]

    report.print_calculation_start(n_steps, settings["dt"])
    return positions, times, n_steps, save_every


def compute_velocity(case):
    """Check the stability and solve the velocity. Returns (times, positions, values)."""
    fluid = FLUIDS[case["fluid"]]
    dt = case["velocity"]["dt"]
    positions, times, n_steps, save_every = make_grid(case, "velocity")

    sigma = get_sigma("velocity", momentum_diffusivity(fluid), dt, case["dy"],
                      case.get("allow_unstable", False))
    gravity_step = dt * gravity_along_plane(case["angle"])
    values = solve_velocity(len(positions), gravity_step, sigma, n_steps, save_every)
    return times, positions, values


def compute_temperature(case):
    """Check the stability and solve the temperature. Returns (times, positions, values)."""
    fluid = FLUIDS[case["fluid"]]
    dt = case["temperature"]["dt"]
    allow_unstable = case.get("allow_unstable", False)
    positions, times, n_steps, save_every = make_grid(case, "temperature")

    sigma_temperature = get_sigma("temperature", thermal_diffusivity(fluid), dt, case["dy"],
                                  allow_unstable)

    # With viscous heating the velocity is also needed. It is solved with its OWN dt
    # (the velocity dt of the case), not with the temperature dt, which can be
    # unstable for the velocity. These values are not used without viscous heating.
    sigma_velocity = 0.0
    gravity_step = 0.0
    velocity_substeps = 1
    if case["use_viscous_heating"]:
        velocity_dt = case["velocity"]["dt"]
        velocity_substeps = get_velocity_substeps(dt, velocity_dt)
        sigma_velocity = get_sigma("velocity", momentum_diffusivity(fluid), velocity_dt,
                                   case["dy"], allow_unstable)
        gravity_step = velocity_dt * gravity_along_plane(case["angle"])
        if velocity_substeps > 1:
            report.print_velocity_substeps(velocity_dt, velocity_substeps,
                                           n_steps * velocity_substeps)

    wall_temperature = case["wall_temperature"]
    surface_temperature = case["surface_temperature"]
    # Initial profile: straight line from the wall (y = 0) to the surface (y = depth).
    initial_profile = (wall_temperature
                       + (surface_temperature - wall_temperature) * positions / case["depth"])

    values = solve_temperature(initial_profile, wall_temperature, sigma_temperature,
                               fluid["viscosity"] / fluid["conductivity"],
                               case["use_viscous_heating"], gravity_step, sigma_velocity,
                               velocity_substeps, n_steps, save_every)
    return times, positions, values


# --- Program -----------------------------------------------------------------
def main():
    report.print_banner()
    case = ask_case()
    fluid = FLUIDS[case["fluid"]]

    report.print_case_title(case)
    report.print_conditions(case, fluid)
    report.print_grid_and_stability(case, fluid)
    reynolds = report.print_analytical(case, fluid)
    if reynolds <= case.get("min_reynolds", 0):
        report.print_stopped(f"the Reynolds number ({report.number(reynolds)}) is not above "
                             f"{case['min_reynolds']}, so this case cannot show what it promises. "
                             f"Change its 'angle' or 'depth' in config.py.")
        return

    # Velocity first, then temperature. All figures are shown together at the end.
    n_plots = 0
    for quantity, compute in (("velocity", compute_velocity),
                              ("temperature", compute_temperature)):
        if case[quantity] is None:
            continue
        report.print_section(quantity.upper())
        start = time.perf_counter()
        try:
            times, positions, values = compute(case)
        except ValueError as error:
            report.print_stopped(error)
            continue
        seconds = time.perf_counter() - start

        report.print_result(case, fluid, quantity, times, positions, values, seconds)
        unit = UNITS[quantity]
        plot_surface(times, positions, values, f"{quantity.capitalize()} ({unit})",
                     f"{quantity.capitalize()}: {case['label']}")
        n_plots += 1

    report.print_plots_message(n_plots)
    show_plots()


if __name__ == "__main__":
    main()
