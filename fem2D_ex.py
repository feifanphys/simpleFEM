import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
from matplotlib import cm
import time


# quadrature points and weights
quad = [(0.0, 1 / 6), (0.5, 4 / 6), (1.0, 1 / 6)]  # simpson rule

# basis functions in 1D; "v" means "value" and "d" means derivative
N1 = {"v": lambda x: x, "d": lambda x: 1}
N2 = {"v": lambda x: 1 - x, "d": lambda x: -1}
Nodal = [N1, N2]

# basis functions in 2D = tensor product of 1D basis functions
class Phi2D:
    def __init__(self, xNodal, yNodal):
        self.xNodal = Nodal[xNodal]
        self.yNodal = Nodal[yNodal]

    # function evaluation; get value of the nodal function phi evaluated at (x, y)
    def v_phi(self, x, y):
        return self.xNodal["v"](x) * self.yNodal["v"](y)

    # derivative; get derivative of the nodal function dphi = (d/dx phi, d/dy phi) evaluated at (x, y)
    def d_phi(self, x, y):
        return np.array(
            [
                self.xNodal["d"](x) * self.yNodal["v"](y),
                self.xNodal["v"](x) * self.yNodal["d"](y),
            ]
        )


# linear basis functions in 2D; 0 means N1 nodal function phi(x) = x, 1 means N2 nodal function phi(x) = 1-x
Nodal2D = {0: Phi2D(0, 0), 1: Phi2D(1, 0), 2: Phi2D(0, 1), 3: Phi2D(1, 1)}


"""
    Create quad element. A quad is basically a bundle of 4 nodes where the 4 nodal basis functions overlap

    (2) -------- (3)
    |             |
    |             |
    |             |
    (0) -------- (1)
    origin has nodal 0  N1(x) * N1(y)  --> Phi2D(0,0)
    right has nodal 1 N2(x) * N1(y)    --> Phi2D(1,0)
    up has nodal 2 N1(x) * N2(y)       --> Phi2D(0,1)
    upright has nodal 3 N2(x) * N2(y)  --> Phi2D(1,1)
"""
class Element:
    def __init__(self, origin, right, up, upRight):
        self.nodes = [origin, right, up, upRight]


class Node:
    def __init__(self, x, y, idx, isBoundary=False):
        self.x, self.y = x, y
        self.idx = idx
        self.isBoundary = isBoundary
    
    def setValue(self, value):
        self.value = value


class Grid:
    def __init__(self, f, xMin=0.0, xMax=1.0, yMin=0.0, yMax=1.0, xpts=11, ypts=11):
        self.xMin, self.xMax = xMin, xMax
        self.yMin, self.yMax = yMin, yMax
        self.hx = (xMax-xMin)/(xpts-1)
        self.hy = (yMax-yMin)/(ypts-1)
        self.xNum = xpts
        self.yNum = ypts
        self.Nodes, self.Elements = [], []
        self.createLUT()
        self.f = f

    def tictoc(func):
        def wrapper(self, *args, **kwargs):
            t1 = time.time()
            result = func(self)
            t2 = time.time() - t1
            print(f'{func.__name__} ran in {t2} seconds')
            return result
        return wrapper

    def setBC(self, bL=None, bR=None, bU=None, bD=None):
        self.bL = bL
        self.bR = bR
        self.bU = bU
        self.bD = bD


    @tictoc
    def createMesh(self):
        # create mesh and transform it into a list of x and y coordinates
        xRange = np.linspace(0, self.xNum-1, self.xNum, dtype=np.int32)
        yRange = np.linspace(0, self.yNum-1, self.yNum, dtype=np.int32)
        self.xCoord, self.yCoord = np.meshgrid(xRange, yRange)
        xList, yList = self.xCoord.ravel(), self.yCoord.ravel()

        # create Nodes; Each node holds a nodal basis function
        for i, (x, y) in enumerate(zip(xList, yList)):
            self.Nodes.append(Node(x, y, idx=i, isBoundary=False))

        # create Dirichlet B.C if assigned
        if self.bL is not None:
            for i, (isBoundary, value) in enumerate(self.bL):
                self.Nodes[i].isBoundary = isBoundary
                self.Nodes[i].setValue(value)

        if self.bR is not None:
            for i, (isBoundary, value) in enumerate(self.bR):
                self.Nodes[self.xNum*(self.yNum-1) + i].isBoundary = isBoundary
                self.Nodes[self.xNum*(self.yNum-1) + i].setValue(value)

        if self.bU is not None:
            for i, (isBoundary, value) in enumerate(self.bU):
                self.Nodes[(i+1)*self.xNum-1].isBoundary = isBoundary
                self.Nodes[(i+1)*self.xNum-1].setValue(value)
        
        if self.bD is not None:
            for i, (isBoundary, value) in enumerate(self.bD):
                self.Nodes[i*self.xNum].isBoundary = isBoundary
                self.Nodes[i*self.xNum].setValue(value)

        # create Elements
        for i, node in enumerate(self.Nodes):
            if node.x < self.xNum-1 and node.y < self.yNum-1:
                self.Elements.append(
                    Element(
                        node,
                        self.Nodes[i + 1],
                        self.Nodes[i + self.xNum],
                        self.Nodes[i + self.xNum + 1],
                    )
                )
    def createLUT(self):
        self.LUT = np.zeros((4,4))
        for i in range(0, 4):
            for j in range(0, 4):
                for x, y, quadWeight in [ (x, y, wX * wY) for (x, wX) in quad for (y, wY) in quad ]:
                    self.LUT[i,j] += quadWeight * Nodal2D[i].d_phi(x, y).dot(Nodal2D[j].d_phi(x, y))

        self.fLUT = np.zeros(4)
        for i in range(0, 4):
            for x, y, quadWeight in [ (x, y, wX * wY) for (x, wX) in quad for (y, wY) in quad] :
                self.fLUT[i] += quadWeight * Nodal2D[i].v_phi(x, y) * self.hx * self.hy

    @tictoc
    def assembleSystem(self):
        # system matrix
        A = dok_matrix((len(self.Nodes), len(self.Nodes)), dtype=np.float32)
        for element in self.Elements:
            for i, node_1 in enumerate(element.nodes):
                for j, node_2 in enumerate(element.nodes):
                    if node_1.isBoundary == True:
                        continue
                    A[node_1.idx, node_2.idx] += self.LUT[i, j]

        # apply homogeneous Dirichlet boundary conditions
        for node in self.Nodes:
            if node.isBoundary == True:
                A[node.idx, node.idx] = 1.0
        return A.tocsr()
    
    @tictoc
    def setRHS(self):
        F = np.zeros(len(self.Nodes), dtype=np.float32)
        for element in self.Elements:
            for i, node in enumerate(element.nodes):
                F[node.idx] += self.fLUT[i] * self.f[node.x, node.y]

        for node in self.Nodes:
            if node.isBoundary == True:
                #set U at Dirichlet boundary = assigned value 
                F[node.idx] = node.value
        return F

    def plotSolutionSurf(self, solution):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(self.xCoord, self.yCoord, solution.reshape(self.yNum, self.xNum), cmap=plt.get_cmap('jet'), rstride=5, cstride=5, antialiased=True)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
    
    def plotSolutionHeatmap(self, solution):
        plt.imshow(
            solution.reshape(self.xNum, self.yNum), extent = [self.yMin, self.yMax, self.xMin, self.xMax],
            cmap="jet")
        plt.colorbar()
        plt.show()



if __name__ == "__main__":
    #Grid resolution
    xpts = 101
    ypts = 101

    #RHS of the Poisson equation
    f = np.zeros((xpts, ypts))
    f[:, :] = -1

    #Define boundary conditions for the 4 edges; Default B.C = Neuman 
    bL = []
    bR = []
    bU = []
    bD = []
    for i in range(0, xpts):
        bL.append((True, 0.0))
        bR.append((True, 0.0))
        bU.append((True, 0.0))
        bD.append((True, 0.0))

    #Setup the system
    grid = Grid(f, xMax=1, yMax=1, xpts=xpts, ypts=ypts)
    grid.setBC(bL=bL, bR=bR, bU=bU, bD=bD)
    grid.createMesh()
    A = grid.assembleSystem()
    F = grid.setRHS()

    #Solve Au = F using direct or iterative methods
    t1 = time.time()
    solution = spsolve(A, F)
    #solution, exit_code = cg(A, F) # U = A^{-1}F
    t2 = time.time() - t1
    print(f"Sparse matrix solve time: {t2} seconds")
    grid.plotSolutionHeatmap(solution)
    
