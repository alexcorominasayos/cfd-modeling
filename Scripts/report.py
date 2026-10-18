"""Everything that is printed in the console.

These functions only print: they do not calculate anything new, so main.py stays
focused on the calculation. The text uses simple symbols so it looks the same in
any console.
"""

import textwrap

import numpy as np

from config import KELVIN_OFFSET, LAMINAR_REYNOLDS_LIMIT, MANY_STEPS, MAX_STABLE_SIGMA
from physics import (analytical_temperature, analytical_velocity, diffusion_number,
                     max_stable_dt, momentum_diffusivity, reynolds_number, thermal_diffusivity)
from solvers import build_grid

WIDTH = 78          # width of the lines and titles
LABEL_WIDTH = 30    # width of the label column in the "label   value" rows


# --- Small helpers ---------------------------------------------------------
def number(value, digits=6):
    """Format a number with a few significant digits (600.0000001 -> '600')."""
    return f"{value:.{digits}g}"


def kelvin_text(kelvin):
    """Temperature in Kelvin followed by Celsius, for example '1473 K (1200 °C)'."""
    if abs(kelvin) >= 1e5:    # the unstable case gives huge numbers: Celsius adds nothing
        return f"{number(kelvin, 5)} K"
    return f"{number(kelvin, 5)} K ({number(kelvin - KELVIN_OFFSET, 5)} °C)"


def print_row(label, value):
    """Print one line of the form '   label ........ value'."""
    print(f"   {label:<{LABEL_WIDTH}}{value}")


def print_section(title):
    """Print a section title with a line, for example '--- FLUID AND CONDITIONS ----'."""
    print()
    print(f"--- {title} " + "-" * (WIDTH - len(title) - 5))


# --- Titles --------------------------------------------------------------------
def print_banner():
    """Print the title of the program (before the questions)."""
    print("=" * WIDTH)
    print(" INCLINED FILM FLOW SIMULATION")
    print("=" * WIDTH)


def print_case_title(case):
    """Print the title of the chosen case."""
    print()
    print("=" * WIDTH)
    print(f" CASE: {case['label']}")
    print("=" * WIDTH)


# --- Information printed before the calculation ----------------------------
def print_conditions(case, fluid):
    """Print the properties of the fluid and the conditions of the case."""
    print_section("FLUID AND CONDITIONS")
    print_row("Fluid", case["fluid"])
    print_row("Density", f"{fluid['density']} kg/m³")
    print_row("Dynamic viscosity", f"{fluid['viscosity']} Pa·s")
    print_row("Specific heat", f"{fluid['specific_heat']} J/(kg·K)")
    print_row("Thermal conductivity", f"{fluid['conductivity']} W/(m·K)")
    print_row("Film depth (yF)", f"{case['depth']} m")
    print_row("Inclination angle", f"{case['angle']}°")

    if case["temperature"] is not None:
        print_row("Viscous heating", "included" if case["use_viscous_heating"] else "not included")
        print_row("Wall temperature", kelvin_text(case["wall_temperature"]) + "  (fixed)")
        print_row("Initial surface temperature", kelvin_text(case["surface_temperature"]))
        print_row("Initial profile", "straight line from the wall to the surface")


def print_grid_and_stability(case, fluid):
    """Print a table with the grid, sigma and the stability of each quantity."""
    print_section("NUMERICAL GRID AND STABILITY")
    print("   The calculation is only stable if sigma <= 0.5, where")
    print("   sigma = diffusivity * dt / dy^2. 'Max dt' is the largest stable time step.")
    print()

    header = (f"   {'Quantity':<13}{'Time (s)':>10}{'dt (s)':>10}{'Steps':>12}{'Cells':>7}"
              f"{'sigma':>10}{'Max dt (s)':>12}   Status")
    print(header)
    print("   " + "-" * (len(header) - 3))

    any_unstable = False
    for quantity, diffusivity in (("velocity", momentum_diffusivity(fluid)),
                                  ("temperature", thermal_diffusivity(fluid))):
        settings = case[quantity]
        if settings is None:
            continue
        dt = settings["dt"]
        sigma = diffusion_number(diffusivity, dt, case["dy"])
        max_dt = max_stable_dt(diffusivity, case["dy"])

        try:
            cells, steps = build_grid(case["depth"], settings["time"], case["dy"], dt)
            cells_text, steps_text = str(cells), f"{steps:,}"
        except ValueError:      # the calculation will stop later with a clear message
            cells_text = steps_text = "invalid"

        if sigma <= MAX_STABLE_SIGMA:
            status = "stable"
        elif case.get("allow_unstable"):
            status = "UNSTABLE (on purpose)"
        else:
            status = "UNSTABLE"
            any_unstable = True

        print(f"   {quantity:<13}{number(settings['time']):>10}{number(dt):>10}{steps_text:>12}"
              f"{cells_text:>7}{sigma:>10.4g}{max_dt:>12.4g}   {status}")

    notes = []
    if case["use_viscous_heating"] and case["temperature"] is not None:
        notes.append("With viscous heating the temperature also needs the velocity, which is")
        notes.append(f"solved with the velocity dt ({number(case['velocity']['dt'])} s).")
    if case.get("allow_unstable"):
        notes.append("WARNING: sigma > 0.5 on purpose. The numerical solution is unstable,")
        notes.append("so its values are not physical.")
    if any_unstable:
        notes.append("WARNING: a quantity is UNSTABLE, so its calculation will be stopped.")
        notes.append("Use a dt below 'Max dt' (or a larger dy) in config.py.")
    if notes:
        print()
        for note in notes:
            print(f"   {note}")


def print_analytical(case, fluid):
    """Print the analytical steady solution, the Reynolds number and the flow regime.

    Returns the Reynolds number.
    """
    print_section("ANALYTICAL REFERENCE (steady state, fully developed flow)")
    depth, angle = case["depth"], case["angle"]
    surface_velocity = analytical_velocity(depth, depth, fluid, angle)
    reynolds = reynolds_number(fluid, surface_velocity, depth)

    print_row("Surface velocity", f"{number(surface_velocity)} m/s")
    print_row("Reynolds number", number(reynolds))
    if reynolds < LAMINAR_REYNOLDS_LIMIT:
        print_row("Flow regime", f"laminar (Re < {LAMINAR_REYNOLDS_LIMIT}): the laminar model is valid")
    else:
        print_row("Flow regime", f"turbulent (Re >= {LAMINAR_REYNOLDS_LIMIT}): the laminar model is not valid")

    # This formula includes viscous heating, so it only describes that case.
    if case["use_viscous_heating"] and case["temperature"] is not None:
        temperature = analytical_temperature(depth, depth, fluid, case["wall_temperature"], angle)
        print_row("Steady surface temperature", kelvin_text(temperature))
    return reynolds


# --- Messages during the calculation ------------------------------------------
def print_calculation_start(n_steps, dt):
    """Print how many time steps are going to be calculated."""
    print(f"   Calculating {n_steps:,} time steps of {number(dt)} s ...")
    if n_steps > MANY_STEPS:
        print("   (That is a lot of steps, it can take a while.)")


def print_velocity_substeps(velocity_dt, substeps, total_steps):
    """Print how the velocity is solved inside the temperature with viscous heating."""
    print(f"   The velocity inside the temperature uses its own dt = {number(velocity_dt)} s:")
    print(f"   {substeps:,} velocity steps per temperature step, {total_steps:,} in total.")
    if total_steps > MANY_STEPS:
        print("   (That is a lot of velocity steps, it can take a while.)")


def print_stopped(message):
    """Print the reason why a calculation was stopped."""
    print(textwrap.fill(f"STOPPED: {message}", width=WIDTH, initial_indent="   ",
                        subsequent_indent="      "))


# --- Results ---------------------------------------------------------------------
def print_result(case, fluid, quantity, times, positions, values, seconds):
    """Print the results of a quantity: the time reached and the values at the surface."""
    print()
    requested = case[quantity]["time"]
    reached = times[-1]
    text = f"{number(reached, 10)} s"
    if abs(reached - requested) > 1e-9 * requested:
        text += f"   (requested {number(requested)} s, not a multiple of dt)"
    print_row("Simulated time", text)

    if quantity == "velocity":
        print_velocity_values(case, fluid, positions, values)
    else:
        print_temperature_values(positions, values)
    print_row("Calculation time", f"{seconds:.1f} s (includes the numba compilation)")


def print_velocity_values(case, fluid, positions, values):
    """Print the velocity at the surface and compare it with the analytical steady value."""
    surface = values[-1, -1]
    print_row("Surface velocity", f"{number(surface)} m/s")

    if case.get("allow_unstable"):
        print("   These values are not physical: the calculation is unstable on purpose.")
        return

    steady = analytical_velocity(case["depth"], case["depth"], fluid, case["angle"])
    print_row("Analytical steady value", f"{number(steady)} m/s")
    print_row("Reached", f"{100 * surface / steady:.1f} % of the steady value")
    print("   (100 % means fully developed flow. Less means the flow is still accelerating)")


def print_temperature_values(positions, values):
    """Print the temperature at the surface at the start and at the end, and the hottest point."""
    final = values[-1]
    # If several nodes share the maximum (the surface node copies its neighbour),
    # we want the highest one, so we search from the surface downwards.
    hottest = len(final) - 1 - int(np.argmax(final[::-1]))
    print_row("Surface temperature at start", kelvin_text(values[0, -1]))
    print_row("Surface temperature at end", kelvin_text(final[-1]))
    print_row("Analytical steady value", f"303 K (30 ºC)")
    print_row("Reached", f"{100 * (1473 - final[-1]) / (1473 - 303):.1f} % of the steady value")
    print("   (100 % means fully developed flow. Less means the flow is still cooling)")


def print_plots_message(n_plots):
    """Print what happens next with the plots."""
    print_section("PLOTS")
    if n_plots == 0:
        print("   Nothing to plot: every calculation was stopped.")
    else:
        print(f"   Opening {n_plots} plot window(s). Close them to finish the program.")
    print()
