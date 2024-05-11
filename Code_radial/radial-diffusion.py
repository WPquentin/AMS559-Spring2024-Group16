#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
RTF = 8.31*300/96485 # RT/(F)


# In[2]:


def phi_pos_eq(w):
    a = [4.558, 1.57, 0.058, 0.621]
    b = [0.154, 0.861, 0.888, 0.941]
    c = [0.749, 0.328, 0.0250, 0.148]
    Up = 0
    for i in range(4):
        Up += a[i] * dl.exp(-((w-b[i])/c[i])**2)
    return Up


def phi_pos_eq_np(w):
    a = [4.558, 1.57, 0.058, 0.621]
    b = [0.154, 0.861, 0.888, 0.941]
    c = [0.749, 0.328, 0.0250, 0.148]
    Up = 0
    for i in range(4):
        Up += a[i] * np.exp(-((w-b[i])/c[i])**2)
    return Up

phi0 = np.linspace(0, 1, 101)
plt.plot(phi0, phi_pos_eq_np(phi0)*RTF)
plt.xlabel("$\\tilde{c}_s$", fontsize=15);
plt.ylabel("$\\tilde{\\phi}_{\\rm eq}$", fontsize=15);


# In[3]:


R0 = 0; R1 = 1; nMesh = 200;
mesh = dl.IntervalMesh(nMesh, R0, R1)
s=3
def denser(x):
    return R1 - (R1-R0)*((R1-x)/(R1-R0))**s
x = mesh.coordinates()
x_bar = denser(x)
mesh.coordinates()[:] = x_bar

P2 = dl.FiniteElement("CG", dl.interval, 2)
P1 = dl.FiniteElement("CG", dl.interval, 1)
#ME = dl.MixedElement([P2, P1, P1])
W  = dl.FunctionSpace(mesh, P1) 


# In[4]:


w = dl.Function(W, name="concentration"); w_ = dl.TestFunction(W); dw = dl.TrialFunction(W)
wn = dl.Function(W)


# In[5]:


class out(dl.SubDomain):
    def inside(self, x, on_boundary):
        return dl.near(x[0], R1, 1e-8)
    
Out = out()
boundaries = dl.MeshFunction("size_t", mesh, 0)
boundaries.set_all(0) #初始全部标记0
Out.mark(boundaries, 1) #mark人工标记
bcs = [dl.DirichletBC(W, 0.0, Out)]
ds = dl.Measure('ds', domain=mesh, subdomain_data=boundaries) #手册P96


# In[6]:


# 三个参数
w0 = 0.4; dPhi = 0.085; k = 100 

# radial cooridnate
R  = dl.SpatialCoordinate(mesh)[0]

# 初始条件
w.vector()[:] = w0; wn.assign(w)

# 估计一下需要的时间步长
eta0 = dPhi - phi_pos_eq(w0)*RTF
dt0 = abs(1/(k*np.sqrt(w0*(1-w0))*(np.exp(eta0/2) - np.exp(-eta0/2))))
# 这个10是我假设的
dt = dl.Constant(dt0/1000)
# 模拟1个单位时间
simulation_steps = int(1000/dt0)

# overpotential
eta = dPhi - phi_pos_eq(w)*RTF
# 弱形式
Res = R**2*((w-wn)*w_ + dt*dl.inner(dl.grad(w), dl.grad(w_)))*dl.dx + dt*k*dl.sqrt(w*(1-w))*(dl.exp(eta/2) - dl.exp(-eta/2))*w_*ds(1)
# Jacobian 
Jac = dl.derivative(Res, w, dw)

# 设置一个上下界保证 w 在 0-1 之间
lower = dl.interpolate(dl.Constant(0.0), W)
upper = dl.interpolate(dl.Constant(1.0), W)

# 求解器的各类参数
snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "mumps",
                                          "relative_tolerance": 1.0e-6,
                                          "absolute_tolerance": 1.0e-6,
                                          "maximum_iterations": 50,
                                          "report": True,
                                          "error_on_nonconvergence": True,
                                          "line_search": "basic"}}
# 设置求解器
problem = dl.NonlinearVariationalProblem(Res, w, [], Jac)
problem.set_bounds(lower, upper)
solver = dl.NonlinearVariationalSolver(problem)
solver.parameters.update(snes_solver_parameters)


# In[7]:


#记录一下右端的浓度值
w_right = [w0]
#求解过程 
for i in range(simulation_steps):
    #dl.solve(Res==0, w)
    solver.solve()
    wn.assign(w)
    w_right.append(w(R1))
    # 设置一下输出


# In[8]:


plt.plot(x_bar, w.compute_vertex_values())


# In[12]:


plt.plot(np.linspace(0, 1, simulation_steps+1), np.array(w_right)-w0)
plt.plot(np.linspace(0, 1, simulation_steps+1), np.linspace(0, 1, simulation_steps+1)**0.5/(dt0))
#plt.yscale("log")
#plt.xscale("log")


# In[ ]:




