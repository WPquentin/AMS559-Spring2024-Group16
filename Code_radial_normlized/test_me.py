### NumericalDiffusionScheme
### Ji Zhang
###
### Outer code used to generate a test run of the diffusion scheme for the case
### with constant diffusivity k(x,t) = k0 and S(x,t) = 0.
### ---------------------------------------------------------------------------

from __future__ import division
import matplotlib as mpl, matplotlib.pyplot as plt
import time
import pickle
import numpy as np
from process_csv_data import read_overpotential_time
import diffusion_scheme as dif, plotting as pl
import argparse

def bp():
    import pdb; pdb.set_trace()

class para_list(object):
    def __init__(self, r_p, D_s, k, c_sm, c_0, tau_diff, tau_react, c_l, path):
        self.r_p = r_p
        self.D_s = D_s
        self.k = k
        self.c_sm = c_sm
        self.c_0 = c_0
        self.tau_diff = tau_diff
        self.tau_react = tau_react
        self.c_l = c_l
        self.k_hat = tau_diff / tau_react
        self.RTF = 8.31*300/96485 # RT/(F)
        self.get_overpotential_time(path)
    
    def get_overpotential_time(self, path):
        # read unnormalized data from csv
        overpotential_data, time_data, selected_row = read_overpotential_time(path)
        # normalization
        self.time_data = time_data / self.tau_diff
        self.overpotential = overpotential_data[selected_row] / self.RTF


# para_list_negative = para_list(r_p=2.5e-6, D_s=5e-15, k=1.1e-11, c_sm=28746.0, c_0=48.86, tau_diff=1.25e3, tau_react=7.81e3, c_l=1000, path='.\overpotential_file\overpotential_neg_charge_1A_Mohtat2020.csv')
para_list_positive = para_list(r_p=3.5e-6, D_s=8e-15, k=5e-11, c_sm=35380.0, c_0=31513.0, tau_diff=1.53e3, tau_react=2.21e3, c_l=1000, path='.\overpotential_file\overpotential_pos_charge_1A_Mohtat2020.csv')


def compute_boundary_condition(c_s = [48.8682], parameter_list = para_list_positive, time_step = 0, const = None):
    if const is not None:
        return const
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
    
    # 3 parameters
    w0 = c_s[-1]; dPhi = 0.085
    k_hat = parameter_list.k_hat
    
    # eta0 = parameter_list.dPhi[time_step] - phi_pos_eq(w0) / RTF
    eta0 = parameter_list.overpotential[time_step]
    print(w0)
    ans = -2 * k_hat*np.sqrt(w0*(1-w0))*(np.exp(eta0/2) - np.exp(-eta0/2))
    return ans



def main(L=1.0, N=60, t_tot=5.0, nt=100, q_max=2.0, para_list = para_list_positive, argConst = None):
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
    D_s = para_list.D_s
    t_tot = para_list.time_data[-1]
    nt = para_list.time_data.shape[0]
    dt = t_tot/nt # time step [s]

    x = np.linspace(h, L, N ) # grid cell centres [m], start with h
    q_init = para_list.c_0 / para_list.c_sm * np.ones_like(x) # initial conditions

    q_old_BE = q_init.copy()
        
    T1 = time.perf_counter()
    q_BE_history_show = [q_init]
    q_BE_history = [q_init]
    q_BE_next = []
    bc_BE_history = []
    for i in range(nt):
        
        bc_old_BE = 2 * h * compute_boundary_condition(q_old_BE, para_list, i, const=argConst)
        q_new_BE = dif.SolveDiffusionEquation(q_old_BE, bc_old_BE, bc_old_BE, dt, h, L, theta = 1.0) # Backward-Euler (BE)
        
        bbcc = compute_boundary_condition(q_new_BE, para_list, i, const=argConst)
        bc_new_BE = 2 * h * bbcc
        q_new_BE = dif.SolveDiffusionEquation(q_old_BE, bc_old_BE, bc_new_BE, dt, h, L, theta = 1.0) # Backward-Euler (BE)
        print("difference between bc_new_BE and bc_new_BE:", bc_new_BE- bc_old_BE)

        q_old_BE = q_new_BE.copy()
        
        if i % 10 == 0:
            q_BE_history_show.append(q_new_BE)
        
        q_BE_history.append(q_new_BE)
        q_BE_next.append(q_new_BE)
        bc_BE_history.append(bbcc)

    T2 = time.perf_counter()

    q_BE_history.pop()
    q_BE_history = np.array(q_BE_history)
    q_BE_next = np.array(q_BE_next)
    bc_BE_history = np.array(bc_BE_history)
    x_dict = {'u':q_BE_history, 'bc':bc_BE_history}
    data_dict = {'x':x_dict, 'y':q_BE_next}

    if argConst is not None:
        label = str(argConst).split('.')[0] + str(argConst).split('.')[1]
        path_name = 'data_dict_constBC_'+ label + '_long.pickle'
        with open(path_name, 'wb') as f:
            pickle.dump(data_dict, f)
    else:
        with open('data_dict_posi.pickle', 'wb') as f:
            pickle.dump(data_dict, f) 

    fig1, ax1 = pl.MakePlots_history(
        x, q_BE_history_show, nt, dt, D_s,
        xlim=[0,L], ylim=[0,q_max])
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    fig1.show()
    # input()

    print("running time: %s ms" % ((T2-T1)*1000))

    # fig2, ax2 = pl.EnergyPlot(
    #     np.linspace(0, nt*dt, nt), int_q_CN, int_q_BE,
    #     xlim=[0,L], ylim=[5,6])
    # fig2.show()


if __name__ == '__main__':
    pl.SetRCParams()
    parser = argparse.ArgumentParser(description='Simulation of radial diffusion equation.')
    
    parser.add_argument('-const', '--const', type=float, default=None, help='constant boundary conditon')
    parser.add_argument('-nt', '--nt', type=int, default=100, help='length of timestep')
    parser.add_argument('-N', '--N', type=int, default=60, help='length of timestep')

    args = parser.parse_args()
    argConst = args.const
    argNt = args.nt
    argN = args.N
    main(argConst = argConst, nt = argNt, N = argN)
