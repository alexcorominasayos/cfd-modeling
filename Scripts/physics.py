"""Physics formulas: analytical solutions, Reynolds number and the stability check.

Every function receives the fluid as a dictionary from config.FLUIDS.
"""

import numpy as np

from config import GRAVITY, MAX_STABLE_SIGMA


def gravity_along_plane(angle):
    """Part of gravity that pushes the fluid down the slope: g * sin(angle).

    The angle is in degrees and depends on the fluid (see config.INCLINATION_DEGREES).
    """
    return GRAVITY * np.sin(np.radians(angle))


# --- Analytical (steady state) solutions -----------------------------------
def analytical_velocity(y, depth, fluid, angle):
    """Velocity at height y, with no slip at the wall and no shear at the surface."""
    factor = fluid["density"] * gravity_along_plane(angle) / fluid["viscosity"]
    return factor * (depth * y - y**2 / 2)


def analytical_temperature(y, depth, fluid, wall_temperature, angle):
    """Temperature at height y when viscous heating is included.

    It comes from integrating k * T'' = -mu * (du/dy)^2 twice, with a fixed wall
    temperature and an insulated free surface (T'(depth) = 0).
    """
    heating = (fluid["density"] * gravity_along_plane(angle)) ** 2 / (
        fluid["conductivity"] * fluid["viscosity"]
    )
    polynomial = depth**2 * y**2 / 2 + y**4 / 12 - y**3 * depth / 3 - depth**3 * y / 3
    return wall_temperature - heating * polynomial


def reynolds_number(fluid, velocity, depth):
    """Reynolds number, using the film depth as the characteristic length."""
    return fluid["density"] * velocity * depth / fluid["viscosity"]


# --- Stability of the explicit scheme --------------------------------------
def momentum_diffusivity(fluid):
    """Kinematic viscosity: how fast velocity spreads across the film (m^2/s)."""
    return fluid["viscosity"] / fluid["density"]


def thermal_diffusivity(fluid):
    """How fast heat spreads across the film (m^2/s)."""
    return fluid["conductivity"] / (fluid["density"] * fluid["specific_heat"])


def diffusion_number(diffusivity, dt, dy):
    """The dimensionless number called sigma: diffusivity * dt / dy^2."""
    return diffusivity * dt / dy**2


def max_stable_dt(diffusivity, dy):
    """Largest time step that keeps sigma under the stability limit."""
    return MAX_STABLE_SIGMA * dy**2 / diffusivity


def get_sigma(field_name, diffusivity, dt, dy, allow_unstable=False):
    """Return sigma, or stop with an error if the scheme would blow up.

    The original scripts only printed the limit and ran anyway, which gave
    meaningless numbers when it was exceeded. Only the cases that want to show
    the instability on purpose can pass allow_unstable=True.
    """
    sigma = diffusion_number(diffusivity, dt, dy)
    if sigma > MAX_STABLE_SIGMA and not allow_unstable:
        raise ValueError(
            f"The {field_name} calculation is unstable (sigma = {sigma:.4g}, the limit "
            f"is {MAX_STABLE_SIGMA}). Use dt <= {max_stable_dt(diffusivity, dy):.4g} s "
            f"or a larger dy."
        )
    return sigma
