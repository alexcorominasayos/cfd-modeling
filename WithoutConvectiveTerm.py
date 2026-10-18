import numpy as np
import matplotlib.pyplot as plt
import numba

#Ferran Alegre, Àlex Coromina, Jordan Soto.
#Code to calculate temperature WITHOUT convective term.

# Constants
g = 9.807
angle = 5  # degrees

# Fluids properties
fluids = {
    "water":     (998, 0.001003, 4182, 0.6, 0.001, 0.45),
    "oil":       (870, 0.104, 1670, 0.144, 0.01, 0.3),
    "glycerin":  (1260, 1.49, 2435, 0.28, 0.05, 1),
    "ammonia":   (608, 0.00022, 2970, 0.48, 0.0005, 0.3),
    "lava":      (2600, 100, 1450, 1.3, 0.01, 100),
}

# Domain
y0, yF = 0, 0.2
t0, tF = 0, 200000

# User Input
def select_fluid():
    choice = input("Choose fluid ('water', 'oil', 'glycerin', 'ammonia', 'lava'): ").strip().lower()
    if choice not in fluids:
        print("Invalid fluid.")
        quit()
    return fluids[choice]

rho, mu, cv, k, dy, dt = select_fluid()
print(f'Lava with rho={rho} kg/m³, mu={mu} Pa·s, cv={cv} J/(kg·K), k={k} W/(m·K), Δy={dy} m, Δt={dt} s.')

# velocity
sigma = (mu * dt) / (rho * dy ** 2)
deltaT = (rho * dy ** 2) / (mu * 2)
print(f'For velocity: sigma ={sigma}, Δt max={deltaT}s')

# temperature
sigma_t = (k * dt) / (rho * cv * dy ** 2)
deltaTt = (rho * cv * dy ** 2) / (2 * k)
print(f'For temperature: sigma={sigma_t}, Δt max={deltaTt}s')


#Reynolds and analitical solutions.
def Reynolds(rho,mu,velocity,yF):
  reynolds = rho*velocity*yF/mu
  return reynolds

def Analitical_Solution_Velocity(g,angle,mu,rho,yf,y):
  velocity_u = g*np.sin(angle*np.pi/180)*rho/mu * (yf*y-y**2/2)
  return velocity_u

def Analitical_Solution_Temperature(g,angle,mu,rho,yF,y):
  A = (rho*g*np.sin(angle*np.pi/180))**2/(k*mu)
  T_wall = 30+273
  temperature = -A*(yF**2*y**2/2+y**4/12 - y**3*yF/3 - yF**3*y/3) + T_wall
  return temperature

y=yF
u = Analitical_Solution_Velocity(g,angle,mu,rho,yF,y)
print(f'The analytical  solution for Velocity the surface, at {yF}m, is {u}m/s')

reynolds = Reynolds(rho,mu,u,yF)
print(f'Reynolds is: {reynolds}')
if reynolds < 1000:
  print('We are in laminar flow, Re < 1000')
else:
  print('We are in turbulent flow, Re > 1000')

T = Analitical_Solution_Temperature(g,angle,mu,rho,yF,y)
print(f'The analytical  solution for Temperature at the surface, at {yF}m, is: {T}ºK')



# Discretization
n = int((tF - t0) / dt)
m = int((yF - y0) / dy)
t = np.linspace(t0, tF, n + 1,)
y = np.linspace(y0, yF, m + 1,)





#--------------------------- Velocity Solver ----------------------------------
@numba.njit(parallel=True)
def compute_velocity_matrix(g, angle, mu, rho, dy, dt, n, m):
    sigma = (mu * dt) / (rho * dy ** 2)
    M = np.zeros((n + 1, m + 1), dtype=np.float64)
    auxM = np.zeros_like(M)

    for _ in range(3000):
        for j in numba.prange(1, m):  # Parallelize outer spatial loop
            for i in range(n):
                M[i+1, j] = dt * g * np.sin(np.radians(angle)) + (1 - 2 * sigma) * auxM[i, j] + sigma * (
                    auxM[i, j + 1] + auxM[i, j - 1])
        for i in numba.prange(n):  # Parallelize this as well
            M[i + 1, m] = M[i + 1, m - 1]
        auxM[:, :] = M
    return M.T


#--------------------------- Temperature Solver ----------------------------------
@numba.njit(parallel=True)
def compute_temperature_matrix(rho, cv, k, mu, dy, dt, n, m):
    sigma = (k * dt) / (rho * cv * dy ** 2)
    M = np.zeros((n + 1, m + 1), dtype=np.float64)
    T_wall = 30 + 273
    T_surface = 1200 + 273

    for i in numba.prange(n + 1):
        M[i, 0] = T_wall
    for j in numba.prange(m + 1):
        y_pos = y0 + j * dy
        M[0, j] = T_surface + (T_wall - T_surface) * (1 - y_pos / yF)

    auxM = M.copy()

    for _ in range(5000):
        for j in numba.prange(1, m):
            for i in range(n):
                viscous_term = 0
                M[i+1, j] = sigma * (auxM[i, j+1] - 2 * auxM[i, j] + auxM[i, j-1] + viscous_term) + M[i, j]
        for i in numba.prange(n):
            M[i+1, m] = M[i+1, m-1]
        auxM[:, :] = M

    return M.T


#--------------------------- Plotting ----------------------------------

def plot_3d_matrix(M, label, zlabel):
    T, Y = np.meshgrid(t, y)
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(121, projection='3d', label=label)
    ax.plot_surface(T, Y, M, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_xlabel('$Time(s)$', fontsize=20, rotation=30)
    ax.set_ylabel('$Position(m)$', fontsize=20, rotation=150)
    ax.set_zlabel(f'${zlabel}$', fontsize=30, rotation=60)
    plt.subplots_adjust(top=0.92, bottom=0.28, left=0.10, right=1.65, hspace=0.25, wspace=0.45)
    print(f'The final value at t={tF} s and y={yF} m is {M[-1][-1]} units')
    plt.show()

#--------------------------- Execution ----------------------------------

choice = input("Introduce 'V' for velocity or 'T' for temperature: ").strip().upper()
if choice == "V":
    Vmatrix = compute_velocity_matrix(g, angle, mu, rho, dy, dt, n, m)
    plot_3d_matrix(Vmatrix, label="Velocity", zlabel="Velocity (m/s)")
elif choice == "T":
    Tmatrix = compute_temperature_matrix(rho, cv, k, mu, dy, dt, n, m)
    plot_3d_matrix(Tmatrix, label="Temperature", zlabel="Temperature (K)")
else:
    print("Invalid option.")
