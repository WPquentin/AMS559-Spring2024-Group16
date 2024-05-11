### NumericalDiffusionScheme
### Jake Aylmer
###
### This code contains two sub-routines needed to solve the diffusion equation
### with variable diffusivity and non-zero source term. It does not include
### advection terms (dq/dx). SolveDiffusionEquation() integrates forward by one
### time-step using SchemeMatrix() to calculate the diffusion operator, A.
### 
### See the repository documentation for further details.
### ---------------------------------------------------------------------------

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def SchemeMatrix(N, D_s, h):
    """Calculate the matrix A which expresses the diffusion operator in the
    numerical scheme for solving the diffusion equation with spatially-variable
    diffusivity u(x)
    
    --Args--
    N   : integer; number of grid cells.
    k   : function; of x, which should return the diffusivity at x.
    h: float, step size.
    """
    A = np.zeros( (N, N) )
    
    # Inner elements (away from boundaries) - all rows except first and last:
    for i in range(1, N-1):
        j = i + 1
        c = 1.0/2
        A[i][i-1] = (j + c)**2 / (j**2)
        A[i][i] = -((j + c)**2 + (j - c)**2) / (j**2)
        A[i][i+1] = (j - c)**2 / (j**2)

    # Neumann boundaries - first and last rows:
    A[0][0] = -2
    A[0][1] = 2
    A[N-1][N-2] = 1 + ((N - 0.5)**2 / (N**2))
    A[N-1][N-1] = - ((N + 0.5)**2 + (N - 0.5)**2) /(N**2)
    
    A *= D_s / (h**2)
    
    return A


def SolveDiffusionEquation(q_old, Boundary_condition_old, Boundary_condition_new, D_s, dt, h, L=1.0, theta=1.0):
    """
    --Args--
    q_old   : NumPy array of length N, q at the current time level.
    Boundary_condition_old   : float
    Boundary_condition_new   : float
    k       : function of x, which should return the diffusivity at x.
    dt      : float, time step.
    (L)     : float, upper limit of spatial domain (i.e. 0 < x < L), default
              L=1.0.
    (theta) : float, between 0 and 1, specifies which scheme is used (0 is
              forward-Euler, 0.5 is Crank-Nicholson, 1 is backward-Euler).
              Default theta=1.
    """
    
    N = len(q_old)
    A = SchemeMatrix(N, D_s, h)
    
    M1 = np.linalg.inv( np.eye(N) - theta*dt*A )
    M2 = np.dot(np.eye(N) + (1-theta)*dt*A, q_old) 
    q_old[-1] += dt*(theta*Boundary_condition_new + (1-theta)*Boundary_condition_old) * D_s / (h**2)

    
    q_new = np.dot(M1, M2) 
    
    return q_new
