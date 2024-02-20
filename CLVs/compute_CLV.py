# Libraries
import numpy as np
from lyapynov import ContinuousDS, CLV

# Definition of Lorenz63.
sigma = 10.
rho = 28.
beta = 8./3.
x0 = np.array([-0.52, -1.0, 6.37])
t0 = 0.
dt = 1e-2

def f(x,t):
    res = np.zeros_like(x)
    res[0] = sigma*(x[1] - x[0])
    res[1] = x[0]*(rho - x[2]) - x[1]
    res[2] = x[0]*x[1] - beta*x[2]
    return res

def jac(x,t):
    res = np.zeros((x.shape[0], x.shape[0]))
    res[0,0], res[0,1] = -sigma, sigma
    res[1,0], res[1,1], res[1,2] = rho - x[2], -1., -x[0]
    res[2,0], res[2,1], res[2,2] = x[1], x[0], -beta
    return res

Lorenz63 = ContinuousDS(x0, t0, f, jac, dt)

# Forward the system.
print("Forward the system.")
Lorenz63.forward(10**6, False)
print("OK")
print("")

# Compute CLV
print("Computation of CLVs.")
CLV, traj = CLV(Lorenz63, 3, 0, 10**5, 10**3, 10**5, True, check = False)
print("OK")
print("")

# Save CLVs and trajectory
print("Save the trajectory and CLVs.")
CLV = np.asarray(CLV)
np.save('./testCLV.npy', CLV)
np.save('./testTraj.npy', traj)
print("OK")