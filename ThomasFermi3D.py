import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from fem3D_ex import *
import pickle

"""
Fermi integral of order 1/2
"""
def FD(x):
    a = np.exp(-x)
    b = 3*(pi/2)**2 * ((x+2.13) + ((abs(x-2.13))**2.4 + 9.6)**(5/12))**-1.5
    return 1/(a+b)

#define P donor density in units of nm^-3
Np_lead = 1.13
Np_bulk = 1E-6
#define EDOS in units of nm^-3
EDOS_lead = 6.0E-5
EDOS_bulk = 1.7E-7
#define dielectric constant
eps_lead = 1
eps_bulk = 11.7
eps0 = 18.06

xDim = 40; yDim = 75; zDim = 40
xpts = 81; ypts = 151; zpts = 81
bD = 0
bU = 0

nd = np.full((xpts, ypts, zpts), Np_bulk)
nd[32:48, 74:77, 0:30] = Np_lead
nd[32:48, 74:77, 50:] = Np_lead
Nc = np.full((xpts, ypts, zpts), EDOS_bulk)
Nc[32:48, 74:77, 0:30] = EDOS_lead
Nc[32:48, 74:77, 50:] = EDOS_lead
eps = np.full((xpts, ypts, zpts), eps0/eps_bulk)
eps[32:48, 74:77, 0:30] = eps0/eps_bulk
eps[32:48, 74:77, 50:] = eps0/eps_bulk

last_ntotal = None #"tj2_200ueV"
build_A = True
output = "tj3_20meV"

maxiter = 200 #max iteration cycle
a = 0.80 #mixing factor between old ntotal and new ntotal
b = 0 #mixing factor between old solution and new solution
conv = 5E-4 #convergence threshold
error = 1 #error between old/new solution
kT = 0.020 #temperature in units of eV


if last_ntotal is None:
    ntotal = nd
else:
    with open(f'{last_ntotal}.pkl', 'rb') as f:
        ntotal = pickle.load(f)

s = np.sum(ntotal)
grid = Grid(eps*ntotal, xMax=xDim, yMax=yDim, zMax=zDim, xpts=xpts, ypts=ypts, zpts=zpts)
grid.setBC(bD=bD, bU = bU)
grid.createMesh()
solution = None
iter = 0

if build_A == True:
    A = grid.assembleSystem()
    with open('A.pkl', 'wb') as f:
        pickle.dump(A, f)
else:
    with open('A.pkl', 'rb') as f:
        A = pickle.load(f)

while error > conv:
    print(f"----------- iteration step {iter} -------------")
    F = grid.setRHS()
    t1 = time.time()
    new_solution, exit_code = minres(A, F, maxiter=200)
    new_solution = new_solution.reshape(xpts, ypts, zpts)
    if solution is None:
        solution = new_solution
    else:
        solution = b * solution + (1 - b) * (new_solution)
    t2 = time.time() - t1
    print(f"Sparse matrix solve time: {t2} seconds")
    #savevtk(solution, [1, 1, 1], f"patch_{i}")
    tmp = (solution-1.13)/kT
    tmp = np.where(tmp>-10, tmp, -10)
    ne = Nc * FD(tmp)
    ntotal = a * ntotal + (1 - a) * (nd-ne)
    grid.f = eps*ntotal
    new_s = np.sum(ntotal)
    error = abs((new_s - s)/s)
    s = new_s
    print(f"total charge: {s}, error = {error}")
    iter += 1
    if iter > maxiter:
        print("Max iteration cycls reached...")
        break;

savevtk(solution, [grid.hx, grid.hy, grid.hz], output)
with open(f'{output}.pkl', 'wb') as f:
    pickle.dump(ntotal, f)




