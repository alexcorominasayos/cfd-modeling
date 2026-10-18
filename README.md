# Inclined Film Flow

Simulation of a viscous fluid film flowing down an inclined plane. The program
calculates how the **velocity** and the **temperature** change across the depth
of the film over time, and shows both as 3D surfaces.

Authors: Ferran Alegre, Àlex Coromina, Jordan Soto.

## The model

`y` is the height above the wall, `depth` is the thickness of the film and the
plane has an inclination angle that depends on the fluid. Both quantities follow
a diffusion equation:

```
du/dt = g·sin(angle) + (mu/rho) · d²u/dy²                         velocity
dT/dt = k/(rho·cv) · d²T/dy² + mu/(rho·cv) · (du/dy)²             temperature
```

| | Velocity | Temperature |
|---|---|---|
| Wall (y = 0) | u = 0 (no slip) | fixed temperature |
| Free surface (y = depth) | no shear | insulated |

The last term of the temperature equation is the **viscous heating** (the heat
produced by friction between layers of fluid). It was called "convective term"
in the first versions of the scripts. The temperature can be calculated with it
or without it.

## How to run

```bash
pip install -r requirements.txt
python main.py
```

Example:

```
Choose fluid (water, oil, glycerin, ammonia, lava): oil
Include the viscous heating term? (y/n): y
```

The program prints a report in sections, then calculates **both** the velocity and
the temperature and opens two 3D plots together:

| Section | What it tells you |
|---|---|
| FLUID AND CONDITIONS | Properties of the fluid, depth, inclination angle and temperatures |
| NUMERICAL GRID AND STABILITY | A table with the time, dt, steps, cells, sigma, the maximum stable dt and whether each quantity is stable |
| ANALYTICAL REFERENCE | The steady-state surface velocity, the Reynolds number and whether the flow is laminar or turbulent |
| VELOCITY / TEMPERATURE | The simulated time, the values at the surface (compared with the analytical value for the velocity), and the calculation time |
| PLOTS | How many plot windows are opened |

If a quantity is unstable, its section says `STOPPED` and explains which `dt` to
use. The text is produced by `report.py`.

Water and lava also ask: `Show the special case ... instead of the normal one? (y/n)`.

## Time shown in each plot

| Fluid | Velocity | Temperature |
|---|---|---|
| water | 600 s | 600 s |
| oil | 300 s | 100000 s |
| glycerin | 40 s | 100000 s |
| ammonia | 600 s | 100 s |
| lava | 10 s | 200000 s |

These times are in `VELOCITY_TIME` and `TEMPERATURE_TIME` in `config.py`.

## Depth of the film (yF)

| Fluid | Depth |
|---|---|
| water | 0.01 m |
| oil, glycerin | 0.1 m |
| ammonia | 0.005 m |
| lava | 0.3 m |

The depths are in `DEPTH` in `config.py`. The depth must be a whole number of
cells (`depth / dy` must be an integer); the program checks it.

## Inclination angle

| Fluid | Angle |
|---|---|
| water, ammonia | 0.01° |
| oil, glycerin | 0.1° |
| lava | 5° |

The angles are in `INCLINATION_DEGREES` in `config.py`. Each case carries its own
angle, so a special case can use a different one.

## Special cases

| Case | What it shows | Time |
|---|---|---|
| Water, Reynolds > 1000 | Velocity only, with an angle of 0.1° (with the normal 0.01° the Reynolds number of water is only 847, which is laminar). The program checks that the Reynolds number is above 1000 before calculating | 10000 s |
| Lava, sigma > 0.5 | Velocity and temperature, **unstable on purpose** (dy = 0.01 m, dt = 250 s), to see how the numerical solution blows up. Viscous heating is off | 10000 s |

Their parameters are in `WATER_HIGH_REYNOLDS` and `LAVA_UNSTABLE` in `config.py`.

## Files

| File | What it does |
|---|---|
| `main.py` | Runs the program: asks, calculates both quantities and plots them |
| `config.py` | Constants, fluids, depths, angles, times, the two scenarios and the special cases (the file you edit to change data) |
| `user_input.py` | The questions to the user, with validation |
| `physics.py` | Analytical solutions, Reynolds number and the stability check |
| `solvers.py` | The grid and the numba solvers (velocity and temperature) |
| `report.py` | Everything that is printed in the console |
| `plotting.py` | The 3D plots |

A **case** is a dictionary with the fluid, geometry (depth and angle), temperatures and, for each
quantity, its time and dt (or `None` if it is not calculated). `main.py` only
reads that dictionary, so a new case needs no changes in the code.

To add a fluid, add it to `FLUIDS`, give it a depth in `DEPTH`, an angle in
`INCLINATION_DEGREES`, a time in `VELOCITY_TIME` and `TEMPERATURE_TIME`, and a
`(dy, velocity dt, temperature dt)` tuple in **both** scenarios.

## How the solvers work

The solvers use explicit finite differences: the profile at the next time step
is calculated from the profile at the current step, node by node. This is
only stable if `sigma = diffusivity · dt / dy² ≤ 0.5`. If a combination of
`dy` and `dt` breaks this limit, the program stops that calculation with an
error that says the maximum `dt` to use (except in the special unstable case).

With viscous heating, the velocity is calculated in the same time loop as the
temperature. The velocity does not depend on the temperature, so the result is
the same as solving it first, but the velocity of every step does not have to be
stored. Only about 1000 time steps of the result are kept (`MAX_SAVED_STEPS` in
`config.py`); the first and the last step are always kept, so the printed final
values are exactly at the end of the simulated time.

The velocity and the temperature can have different time steps, because each one
must satisfy its own stability limit. For example lava needs 0.001 s for the
velocity but its temperature can use 100 s. With viscous heating the velocity
inside the temperature calculation keeps its own dt: after each temperature step
it advances `temperature dt / velocity dt` steps (100000 for lava), so both always
describe the same instant. For this the temperature dt must be a whole multiple of
the velocity dt, and the program checks it. The temperature step uses the velocity
of the beginning of the step, as in the first scripts.

## Things to keep in mind

* Lava with viscous heating needs 2·10⁸ velocity steps for 200000 s (2000
  temperature steps of 100 s, with 100000 velocity steps each). With compiled
  code it should take several seconds. The program warns above 5·10⁷ steps.
* The free surface condition copies the neighbour node, which is only first-order
  accurate. The steady velocity at the surface is lower than the analytical one
  by about 1 / (number of cells): 10 % with 10 cells, 2 % with 50 cells. A
  smaller `dy` improves it, but `dt` must be about four times smaller each time
  `dy` is halved, to keep sigma <= 0.5.
* The scenario without viscous heating uses 1200 °C at the surface for every
  fluid, including water and ammonia.

## Contributors

* **Ferran Alegre Baltasar** - [@ferran-alegre](https://github.com/ferran-alegre)
* **Àlex Coromina Sayós** - [@alexcorominasayos](https://github.com/alexcorominasayos)
* **Jordán Frank Soto Huarcaya** - [@jordanfranksoto-netizen](https://github.com/jordanfranksoto-netizen)