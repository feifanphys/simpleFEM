import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg, minres
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from np2vtk import savevtk


# quadrature points and weights
quad = [(0.0, 1 / 6), (0.5, 4 / 6), (1.0, 1 / 6)]  # simpson rule

# basis functions in 1D; "v" means "value" and "d" means derivative
N1 = {"v": lambda x: x, "d": lambda x: 1}
N2 = {"v": lambda x: 1 - x, "d": lambda x: -1}
Nodal = [N1, N2]

# basis functions in 3D = tensor product of 1D basis functions
class Phi3D:
    def __init__(self, xNodal, yNodal, zNodal):
        self.xNodal = Nodal[xNodal]
        self.yNodal = Nodal[yNodal]
        self.zNodal = Nodal[zNodal]

    # function evaluation; get value of the nodal function phi evaluated at (x, y, z)
    def v_phi(self, x, y, z):
        return self.xNodal["v"](x) * self.yNodal["v"](y) * self.zNodal["v"](z)

    # derivative; get derivative of the nodal function dphi = (d/dx phi, d/dy phi) evaluated at (x, y)
    def d_phi(self, x, y, z):
        return np.array(
            [
                self.xNodal["d"](x) * self.yNodal["v"](y) * self.zNodal["v"](z),
                self.xNodal["v"](x) * self.yNodal["d"](y) * self.zNodal["v"](z),
                self.xNodal["v"](x) * self.yNodal["v"](y) * self.zNodal["d"](z)
            ]
        )


# linear basis functions in 3D; 0 means N1 nodal function phi(x) = x, 1 means N2 nodal function phi(x) = 1-x
Nodal3D = {0: Phi3D(0, 0, 0), 1: Phi3D(1, 0, 0), 2: Phi3D(0, 1, 0), 3: Phi3D(1, 1, 0),
           4: Phi3D(0, 0, 1), 5: Phi3D(1, 0, 1), 6: Phi3D(0, 1, 1), 7: Phi3D(1, 1, 1)}



class Element:
    def __init__(self, z000, z100, z010, z110, z001, z101, z011, z111):
        self.nodes = [z000, z100, z010, z110, z001, z101, z011, z111]


class Node:
    def __init__(self, x, y, z, idx, isBoundary=False):
        self.x, self.y, self.z = x, y, z
        self.idx = idx
        self.isBoundary = isBoundary
    
    def setValue(self, value):
        self.value = value


class Grid:
    def __init__(self, f, xMin=0.0, xMax=1.0, yMin=0.0, yMax=1.0, zMin=0.0, zMax=1.0, xpts=11, ypts=11, zpts=11):
        self.xMin, self.xMax = xMin, xMax
        self.yMin, self.yMax = yMin, yMax
        self.zMin, self.zMax = zMin, zMax
        self.hx = (xMax-xMin)/(xpts-1)
        self.hy = (yMax-yMin)/(ypts-1)
        self.hz = (zMax-zMin)/(zpts-1)
        self.xNum = xpts
        self.yNum = ypts
        self.zNum = zpts
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

    def setBC(self, bL=None, bR=None, bU=None, bD=None, bF=None, bB=None):
        self.bL = bL
        self.bR = bR
        self.bU = bU
        self.bD = bD
        self.bF = bF
        self.bB = bB


    @tictoc
    def createMesh(self):
        # create mesh and transform it into a list of x and y coordinates
        self.xRange = np.linspace(0, self.xNum-1, self.xNum, dtype=np.int32)
        self.yRange = np.linspace(0, self.yNum-1, self.yNum, dtype=np.int32)
        self.zRange = np.linspace(0, self.zNum-1, self.zNum, dtype=np.int32)
        self.yCoord, self.xCoord, self.zCoord = np.meshgrid(self.yRange, self.xRange, self.zRange)
        xList, yList, zList = self.xCoord.ravel(), self.yCoord.ravel(), self.zCoord.ravel()

        # create Nodes; Each node holds a nodal basis function
        for i, (x, y, z) in enumerate(zip(xList, yList, zList)):
            self.Nodes.append(Node(x, y, z, idx=i, isBoundary=False))

        # assign Dirichlet if assigned
        if self.bD is not None:
            for i in range(0, self.yNum):
                for j in range(0, self.zNum):
                    self.Nodes[i*self.zNum + j].isBoundary = True
                    self.Nodes[i*self.zNum + j].setValue(self.bD)

        if self.bU is not None:
            for i in range(0, self.yNum):
                for j in range(0, self.zNum):
                    self.Nodes[(self.xNum-1)*self.yNum*self.zNum + i*self.zNum + j].isBoundary = True
                    self.Nodes[(self.xNum-1)*self.yNum*self.zNum + i*self.zNum + j].setValue(self.bU)


        # create Elements
        for i, node in enumerate(self.Nodes):
            if node.x < self.xNum-1 and node.y < self.yNum-1 and node.z < self.zNum-1:
                self.Elements.append(
                    Element(
                        node,
                        self.Nodes[i + self.yNum*self.zNum],
                        self.Nodes[i + self.zNum],
                        self.Nodes[i + (self.yNum+1)*self.zNum],
                        self.Nodes[i + 1],
                        self.Nodes[i + self.yNum*self.zNum + 1],
                        self.Nodes[i + self.zNum + 1],
                        self.Nodes[i + (self.yNum+1)*self.zNum + 1]
                    )
                )
    def createLUT(self):
        self.LUT = np.zeros((8,8))
        for i in range(0, 8):
            for j in range(0, 8):
                for x, y, z, quadWeight in [ (x, y, z, wX * wY * wZ) for (x, wX) in quad for (y, wY) in quad for (z, wZ) in quad]:
                    self.LUT[i,j] += quadWeight * Nodal3D[i].d_phi(x, y, z).dot(Nodal3D[j].d_phi(x, y, z))
        
        self.fLUT = np.zeros(8)
        for i in range(0, 8):
            for x, y, z, quadWeight in [ (x, y, z, wX * wY * wZ) for (x, wX) in quad for (y, wY) in quad for (z, wZ) in quad]:
                self.fLUT[i] += quadWeight * Nodal3D[i].v_phi(x, y, z) * self.hx * self.hy * self.hz



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
                F[node.idx] += self.fLUT[i] * self.f[node.x, node.y, node.z]

        for node in self.Nodes:
            if node.isBoundary == True:
                #set U at Dirichlet boundary = assigned value 
                F[node.idx] = node.value
        return F
    


if __name__ == "__main__":
    #Grid resolution
    xpts = 51
    ypts = 51
    zpts = 51

    #RHS of the Poisson equation
    f = np.zeros((xpts, ypts, zpts))
    f[25:28, 25:28, 25:26] = 1.13*1.544

    #Define boundary conditions for the 4 edges; Default B.C = Neuman 
    bD= 0
    bU = 0


    #Setup the system
    grid = Grid(f, xMax=100, yMax=100, zMax=100, xpts=xpts, ypts=ypts, zpts=zpts)
    grid.setBC(bD=bD, bU = bU)
    grid.createMesh()
    A = grid.assembleSystem()
    F = grid.setRHS()

    #Solve Au = F using direct or iterative methods
    t1 = time.time()
    #solution = spsolve(A, F)
    solution, exit_code = minres(A, F, maxiter=200)
    solution = solution.reshape(xpts, ypts, zpts)
    t2 = time.time() - t1
    print(f"Sparse matrix solve time: {t2} seconds")
    savevtk(solution, [1, 1, 1], "singleDopant")
    #grid.plotSolutionHeatmap(solution[:,:,5])
    
