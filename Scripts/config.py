"""Fixed data of the project: constants, fluids, scenarios and the cases that can be run.

A "case" is a dictionary that describes everything one run needs: the fluid, the
geometry (depth and inclination angle), the temperatures and, for velocity and
temperature, the simulated time and the time step. If a quantity is None, it is
not calculated in that case.
"""

# --- Constants -------------------------------------------------------------
GRAVITY = 9.807            # m/s^2
KELVIN_OFFSET = 273        # Celsius -> Kelvin (same value as the original scripts)
LAMINAR_REYNOLDS_LIMIT = 1000
MAX_STABLE_SIGMA = 0.5     # the explicit scheme only works if sigma <= 0.5
MAX_SAVED_STEPS = 1000     # keeps the results matrix small in very long simulations
MANY_STEPS = 50000000    # above this number of steps the program warns about the wait

# --- Fluids (SI units) -----------------------------------------------------
FLUIDS = {
    "water":    {"density": 998,  "viscosity": 0.001003, "specific_heat": 4182, "conductivity": 0.6},
    "oil":      {"density": 870,  "viscosity": 0.104,    "specific_heat": 1670, "conductivity": 0.144},
    "glycerin": {"density": 1260, "viscosity": 1.49,     "specific_heat": 2435, "conductivity": 0.28},
    "ammonia":  {"density": 608,  "viscosity": 0.00022,  "specific_heat": 2970, "conductivity": 0.48},
    "lava":     {"density": 2600, "viscosity": 100,      "specific_heat": 1450, "conductivity": 1.3},
}

# --- Depth of the film, yF in the original scripts (m) ---------------------
# It must be a whole number of cells: depth / dy has to be an integer in every case.
DEPTH = {"water": 0.01, "oil": 0.1, "glycerin": 0.1, "ammonia": 0.005, "lava": 0.3}

# --- Inclination angle of the plane (degrees) ------------------------------
INCLINATION_DEGREES = {"water": 0.01, "oil": 0.1, "glycerin": 0.1, "ammonia": 0.01, "lava": 5}

# --- Time shown in each plot (seconds) -------------------------------------
VELOCITY_TIME = {"water": 600, "oil": 300, "glycerin": 40, "ammonia": 300, "lava": 10}
TEMPERATURE_TIME = {"water": 600, "oil": 100000, "glycerin": 100000, "ammonia": 100, "lava": 200000}

# --- Scenarios -------------------------------------------------------------
# "Viscous heating" is the term called "convective term" in the original scripts:
# it is the heat produced by the friction between layers of fluid.
#
# "grid" gives, for each fluid: (dy in m, dt for velocity in s, dt for temperature in s).
# Two different dt are needed because, without viscous heating, the temperature
# does not depend on the velocity and can use a much larger dt (lava: 100 s),
# but the velocity of lava needs a tiny one (0.001 s) to be stable.
# With viscous heating the temperature also needs the velocity, so the velocity is
# solved inside the temperature loop, but with ITS OWN dt: the temperature dt must
# be a whole multiple of the velocity dt (lava: 100 / 0.001 = 100000 velocity
# steps for each temperature step).
#
# The error of the steady velocity at the surface is about 1/cells (2 % with 50
# cells, 10 % with 10 cells), and halving dy needs a dt four times smaller to
# keep sigma <= 0.5.

# Old WithConvectiveTerm.py
WITH_VISCOUS_HEATING = {
    "use_viscous_heating": True,
    "wall_temperature": 25 + KELVIN_OFFSET,
    "surface_temperature": 1100 + KELVIN_OFFSET,    # initial temperature at the top
    "grid": {
        "water":    (0.0005, 0.1, 0.1),
        "oil":      (0.005, 0.1, 0.1),
        "glycerin": (0.005, 0.01, 0.01),
        "ammonia":  (0.00025, 0.002, 0.002),
        "lava":     (0.01, 0.001, 100),
    },
}

# Old WithoutConvectiveTerm.py
WITHOUT_VISCOUS_HEATING = {
    "use_viscous_heating": False,
    "wall_temperature": 30 + KELVIN_OFFSET,
    "surface_temperature": 1200 + KELVIN_OFFSET,
    "grid": {
        "water":    (0.0005, 0.1, 0.1),
        "oil":      (0.005, 0.1, 0.1),
        "glycerin": (0.005, 0.01, 0.01),
        "ammonia":  (0.00025, 0.002, 0.002),
        "lava":     (0.01, 0.001, 100),
    },
}


def make_normal_case(fluid_name, use_viscous_heating):
    """Build the normal case of a fluid, with or without viscous heating."""
    scenario = WITH_VISCOUS_HEATING if use_viscous_heating else WITHOUT_VISCOUS_HEATING
    dy, velocity_dt, temperature_dt = scenario["grid"][fluid_name]
    heating_text = "with" if use_viscous_heating else "without"
    return {
        "label": f"{fluid_name}, {heating_text} viscous heating",
        "fluid": fluid_name,
        "depth": DEPTH[fluid_name],
        "angle": INCLINATION_DEGREES[fluid_name],
        "wall_temperature": scenario["wall_temperature"],
        "surface_temperature": scenario["surface_temperature"],
        "use_viscous_heating": use_viscous_heating,
        "dy": dy,
        "velocity": {"time": VELOCITY_TIME[fluid_name], "dt": velocity_dt},
        "temperature": {"time": TEMPERATURE_TIME[fluid_name], "dt": temperature_dt},
    }


# --- Special cases (one per fluid, optional) -------------------------------
# Water with a Reynolds number above 1000 (turbulent regime), only the velocity.
# With the normal angle of water (0.01 degrees) its Reynolds number is only 847, which
# is laminar. An angle of 0.1 degrees (the one of the other fluids) gives 8473, and
# the program checks that it is really above "min_reynolds" before calculating.
WATER_HIGH_REYNOLDS = {
    "label": "water with Reynolds > 1000",
    "fluid": "water",
    "depth": DEPTH["water"],
    "angle": 0.1,
    "use_viscous_heating": False,
    "min_reynolds": LAMINAR_REYNOLDS_LIMIT,
    "dy": 0.0002,
    "velocity": {"time": 10000, "dt": 0.016},
    "temperature": None,
}

# Lava with sigma > 0.5: the explicit scheme is unstable ON PURPOSE, to see how
# the numerical solution blows up. With dy = 0.01 m, dt = 250 s gives sigma = 0.86
# for the temperature and sigma = 96154 for the velocity.
# Viscous heating is off: the velocity grows so much that its square (used by the
# heating term) would overflow and the temperature would be filled with inf.
LAVA_UNSTABLE = {
    "label": "lava with sigma > 0.5 (unstable on purpose)",
    "fluid": "lava",
    "depth": DEPTH["lava"],
    "angle": INCLINATION_DEGREES["lava"],
    "wall_temperature": 30 + KELVIN_OFFSET,
    "surface_temperature": 1200 + KELVIN_OFFSET,
    "use_viscous_heating": False,
    "allow_unstable": True,
    "dy": 0.01,
    "velocity": {"time": 10000, "dt": 250},
    "temperature": {"time": 10000, "dt": 250},
}

SPECIAL_CASES = {"water": WATER_HIGH_REYNOLDS, "lava": LAVA_UNSTABLE}
