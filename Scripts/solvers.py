"""Grid creation and the numerical solvers (explicit finite differences).

The solvers move forward in time, step by step. At every step the new profile is
computed from the previous one, so only two rows per field are kept in memory
while the loop runs. A row is copied into the results matrix only every
`save_every` steps, and the last step is always copied too, so the final result
is exactly the state at the end of the simulated time.

The results have the shape (saved time level, node across the film).
"""

import numba
import numpy as np


def build_grid(depth, final_time, dy, dt):
    """Return (n_cells, n_steps) for the space and time grid.

    The depth must be a whole number of cells. Floats are rounded first because
    a plain int() fails: for example 0.3 / 0.05 is 5.999999999999999 and int()
    would give 5 cells instead of 6.
    """
    n_cells = round(depth / dy)
    if abs(n_cells * dy - depth) > 1e-9 * depth:
        raise ValueError(f"The depth {depth} m is not a multiple of dy = {dy} m.")
    if n_cells < 2:
        raise ValueError(f"dy = {dy} m is too large for a depth of {depth} m.")

    # The final time does not have to be a multiple of dt: we stop at the last full step.
    n_steps = int(final_time / dt + 1e-9)
    if n_steps < 1:
        raise ValueError(f"dt = {dt} s is larger than the final time {final_time} s.")
    return n_cells, n_steps


def saved_steps(n_steps, save_every):
    """Return the steps that the solvers save: 0, save_every, 2*save_every... and the last one."""
    steps = np.arange(0, n_steps + 1, save_every)
    if steps[-1] != n_steps:
        steps = np.append(steps, n_steps)
    return steps


def get_velocity_substeps(temperature_dt, velocity_dt):
    """Return how many velocity steps fit in one temperature step.

    With viscous heating the temperature needs the velocity at each of its time
    levels. The velocity is solved with its own dt (the one that is stable for
    it), so the temperature dt must be a whole multiple of the velocity dt.
    For example lava: 100 s / 0.001 s = 100000 velocity steps per temperature step.
    """
    substeps = round(temperature_dt / velocity_dt)
    if substeps < 1 or abs(substeps * velocity_dt - temperature_dt) > 1e-9 * temperature_dt:
        raise ValueError(
            f"With viscous heating the temperature dt ({temperature_dt} s) must be a "
            f"whole multiple of the velocity dt ({velocity_dt} s)."
        )
    return substeps


# --- One time step (used by both solvers, so the formulas are written only once) ---
@numba.njit
def advance_velocity(old, new, gravity_step, sigma):
    """Calculate the velocity of the next step (new) from the current one (old).

    For each inner node j:
        new[j] = gravity_step + (1 - 2*sigma) * old[j] + sigma * (old[j+1] + old[j-1])

    gravity_step is dt * g * sin(angle): the speed that gravity adds in one step.
    """
    last = len(old) - 1
    for j in range(1, last):
        new[j] = gravity_step + (1 - 2 * sigma) * old[j] + sigma * (old[j + 1] + old[j - 1])
    new[0] = 0.0                # wall: the fluid does not slip
    new[last] = new[last - 1]   # free surface: no shear (zero slope)


@numba.njit
def advance_temperature(old, new, velocity, wall_temperature, sigma, viscous_factor):
    """Calculate the temperature of the next step (new) from the current one (old).

    For each inner node j:
        new[j] = old[j] + sigma * (old[j+1] - 2*old[j] + old[j-1] + heating)
        heating = viscous_factor * (velocity[j+1] - velocity[j])^2

    viscous_factor is viscosity / conductivity. The velocity difference between
    neighbour nodes already includes the factor dy, which is why `heating` has
    units of Kelvin like the other terms inside the brackets.
    If the velocity is all zeros, the heating is zero.
    """
    last = len(old) - 1
    for j in range(1, last):
        velocity_jump = velocity[j + 1] - velocity[j]
        heating = viscous_factor * velocity_jump**2
        new[j] = old[j] + sigma * (old[j + 1] - 2 * old[j] + old[j - 1] + heating)
    new[0] = wall_temperature   # wall: fixed temperature
    new[last] = new[last - 1]   # free surface: insulated (zero slope)


# --- Time loops ---------------------------------------------------------------
@numba.njit
def solve_velocity(n_nodes, gravity_step, sigma, n_steps, save_every):
    """Velocity of the fluid, starting from rest. Returns the saved time levels."""
    n_rows = n_steps // save_every + 1
    if n_steps % save_every != 0:
        n_rows += 1                                  # one more row for the very last step
    saved = np.zeros((n_rows, n_nodes))              # row 0: all zeros (rest)
    old = np.zeros(n_nodes)
    new = np.zeros(n_nodes)

    for step in range(n_steps):
        advance_velocity(old, new, gravity_step, sigma)
        old[:] = new
        if (step + 1) % save_every == 0:
            saved[(step + 1) // save_every, :] = old

    if n_steps % save_every != 0:
        saved[n_rows - 1, :] = old                   # the last step is always saved
    return saved


@numba.njit
def solve_temperature(initial_profile, wall_temperature, sigma_temperature, viscous_factor,
                      use_viscous_heating, gravity_step, sigma_velocity, velocity_substeps,
                      n_steps, save_every):
    """Temperature of the fluid. Returns the saved time levels.

    With viscous heating the velocity is calculated in the same loop. The velocity
    does not depend on the temperature, so the result is the same as using a
    previously solved velocity, but there is no need to store the velocity of
    every step (that would not fit in memory in long runs).

    The velocity uses its OWN time step: gravity_step and sigma_velocity must
    be calculated with the velocity dt, not with the temperature dt. After each
    temperature step, the velocity advances velocity_substeps steps, so both
    always describe the same instant. Without viscous heating the velocity stays
    at zero and is never calculated.
    """
    n_nodes = len(initial_profile)
    n_rows = n_steps // save_every + 1
    if n_steps % save_every != 0:
        n_rows += 1                                  # one more row for the very last step
    saved = np.zeros((n_rows, n_nodes))
    saved[0, :] = initial_profile
    temperature_old = initial_profile.copy()
    temperature_new = initial_profile.copy()
    velocity_old = np.zeros(n_nodes)
    velocity_new = np.zeros(n_nodes)

    for step in range(n_steps):
        # The heating uses the velocity of the current step, so the temperature goes first.
        advance_temperature(temperature_old, temperature_new, velocity_old,
                            wall_temperature, sigma_temperature, viscous_factor)
        if use_viscous_heating:
            for _ in range(velocity_substeps):
                advance_velocity(velocity_old, velocity_new, gravity_step, sigma_velocity)
                velocity_old[:] = velocity_new
        temperature_old[:] = temperature_new

        if (step + 1) % save_every == 0:
            saved[(step + 1) // save_every, :] = temperature_old

    if n_steps % save_every != 0:
        saved[n_rows - 1, :] = temperature_old       # the last step is always saved
    return saved
