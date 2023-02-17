from fem2D_ex import *
import numpy as np

xpts = 501
ypts = 501

nd = np.zeros((xpts, ypts))
nd[100:400, 200:210] = 0.0000001

bL = []
bR = []
for i in range(0, xpts):
    bL.append((True, 0.0))
    bR.append((True, 0.0))

V_0 = np.zeros((xpts, ypts))

n = -0.0000005*(np.exp(V_0)-1) + nd
ntotal = np.sum(n)

grid = Grid(n, xMax=xpts-1, yMax=ypts-1, xpts=xpts, ypts=ypts)
grid.setBC(bL=bL, bR=None)
grid.createMesh()
solution = None
error = 1
iterations = 0
A = grid.assembleSystem()
while error > 1E-4:
    print(f"----------- iteration step {iterations} -------------")
    F = grid.setRHS()
    solution = spsolve(A, F)
    V_0 = solution.reshape(xpts, ypts)
    n = -0.0000005*(np.exp(V_0)-1) + nd
    grid.f = n
    new_ntotal = np.sum(n)
    error = abs((new_ntotal-ntotal)/ntotal)
    print(f"total charge {ntotal} --> {new_ntotal}, error = {error}")
    ntotal = new_ntotal
    iterations += 1 

grid.plotSolutionSurf(solution)


