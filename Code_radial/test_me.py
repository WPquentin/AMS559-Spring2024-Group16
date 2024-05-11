### NumericalDiffusionScheme
### Ji Zhang
###
### Outer code used to generate a test run of the diffusion scheme for the case
### with constant diffusivity k(x,t) = k0 and S(x,t) = 0.
### ---------------------------------------------------------------------------

from __future__ import division
import matplotlib as mpl, matplotlib.pyplot as plt
import time
import numpy as np
import diffusion_scheme as dif, plotting as pl

def ConstantDiffusivity(x, k0=2.5E-3):
    """Return the constant diffusivity at position x."""
    return k0

def AnalyticSolution(x, t, k0=0.25, L=2.0, q_max=4.0, truncate=20):
    """The analytic solution to the diffusion equation with constant
    diffusivity k=k0, zero sources/sinks (S=0), Neumann boundary conditions and
    initial conditions q(x,0) = q_max*x*(L-x). See repository documentation for
    details.
    
    --Arguments (optional ones in round brackets)--
    x          : NumPy array, x-coordinates.
    t          : float, time at which to return the solution.
    (k0)       : float, constant value of diffusivity (default 0.25).
    (q_max)    : float, maximum of initial q profile.
    (truncate) : integer, where to truncate the series expansion.
    """
    q_an = np.zeros(len(x)) + (L**2*q_max/6)
    
    for n in np.arange(2, truncate+1, 2):
        coeff = (-4*q_max*L**2/((n*np.pi)**2))
        q_an += coeff*np.exp(-k0*(n*np.pi/L)**2*t)*np.cos(n*np.pi*x/L)
    
    return q_an

def compute_boundary_condition(c_s = [0.4], k = 1.0, F = 1.0, D_s = 1.0):
    # c_s is an array of solution, k is the scale parameter
    # F is another parameter
    RTF = 8.31*300/96485 # RT/(F)

    def phi_pos_eq(w):
        a = [4.558, 1.57, 0.058, 0.621]
        b = [0.154, 0.861, 0.888, 0.941]
        c = [0.749, 0.328, 0.0250, 0.148]
        Up = 0
        for i in range(4):
            Up += a[i] * np.exp(-((w-b[i])/c[i])**2)
        return Up


    def phi_pos_eq_np(w):
        a = [4.558, 1.57, 0.058, 0.621]
        b = [0.154, 0.861, 0.888, 0.941]
        c = [0.749, 0.328, 0.0250, 0.148]
        Up = 0
        for i in range(4):
            Up += a[i] * np.exp(-((w-b[i])/c[i])**2)
        return Up
    
    # 3 parameters
    w0 = c_s[-1]; dPhi = 0.085; k = 100 
    # c_s_max = max(c_s)
    # c_s_r = c_s[-1]
    # i_loc = k * np.power(c_s_r, 0.5) * np.power(c_s_max, 0.5)
    # ans = i_loc / (F * D_s)

    eta0 = dPhi - phi_pos_eq(w0)*RTF
    ans = k*np.sqrt(w0*(1-w0))*(np.exp(eta0/2) - np.exp(-eta0/2))
    return ans

def main(L=1.0, N=60, t_tot=1.0, nt=60, q_max=1.0):
    """Main routine which is run on start-up of the program. It may be re-run
    with different arguments in a Python interpretter.
    
    --Optional arguments--
    L     : float, upper domain limit in m (so that 0 < x < L).
    N     : int, number of grid cells to split the domain into.
    t_tot : float, total integration time (s).
    nt    : int, number of time steps (this determines the time-step through
            dt = t_tot / nt).
    q_max : float, sets the overall magnitude of the initial conditions.
    """
    h = L/(N+1) # width of grid cells [m]
    D_s = ConstantDiffusivity(0)
    dt = t_tot/nt # time step [s]
    i_loc_old = 1.0 # boundary condition on L
    F_old = 1.0 # boundary condition on L
    i_loc_new = 1.0 # boundary condition on L
    F_new = 1.0 # boundary condition on L

    x = np.linspace(h, L, N ) # grid cell centres [m], start with h
    q_init = 0.4 * np.ones_like(x) # initial conditions
    boundary_cond_old = -2 * h * i_loc_old / (F_old * D_s) # old boundary condition 
    boundary_cond_new = -2 * h * i_loc_new / (F_new * D_s) # new boundary condition 

    # q_an = AnalyticSolution(x, nt*dt, D_s, L, q_max)
    q_old_CN = q_init.copy(); q_old_BE = q_init.copy()
    
    int_q_CN = np.zeros(nt); int_q_BE = np.zeros(nt)
    
    T1 = time.perf_counter()

    for i in range(nt):
        
        int_q_CN[i] = np.sum(q_old_CN) * h
        int_q_BE[i] = np.sum(q_old_BE) * h
        
        bc_old_CN = -2 * h * compute_boundary_condition(q_old_CN)
        bc_old_BE = -2 * h * compute_boundary_condition(q_old_BE)
        q_new_CN = dif.SolveDiffusionEquation(q_old_CN, bc_old_CN, bc_old_CN,
            D_s, dt, h, L, theta = 0.5) # Crank-Nicolson (CN)
        q_new_BE = dif.SolveDiffusionEquation(q_old_BE, bc_old_BE, bc_old_BE,
            D_s, dt, h, L, theta = 1.0) # Backward-Euler (BE)
        bc_new_CN = -2 * h * compute_boundary_condition(q_new_CN)
        bc_new_BE = -2 * h * compute_boundary_condition(q_new_BE)
        q_new_CN = dif.SolveDiffusionEquation(q_old_CN, bc_old_CN, bc_new_CN,
            D_s, dt, h, L, theta = 0.5) # Crank-Nicolson (CN)
        q_new_BE = dif.SolveDiffusionEquation(q_old_BE, bc_old_BE, bc_new_BE,
            D_s, dt, h, L, theta = 1.0) # Backward-Euler (BE)
        print("difference between bc_new_BE and bc_new_BE:", bc_new_BE- bc_old_BE)

        q_old_CN = q_new_CN.copy()
        q_old_BE = q_new_BE.copy()
    
    T2 = time.perf_counter()
    fig1, ax1 = pl.MakePlots(
        x, q_init, q_new_CN, q_new_BE, nt, dt, D_s,
        xlim=[0,L], ylim=[0,q_max])
    fig1.show()

    print("running time: %s ms" % ((T2-T1)*1000))

    # fig2, ax2 = pl.EnergyPlot(
    #     np.linspace(0, nt*dt, nt), int_q_CN, int_q_BE,
    #     xlim=[0,L], ylim=[5,6])
    # fig2.show()
    input()


if __name__ == '__main__':
    pl.SetRCParams()
    main()
