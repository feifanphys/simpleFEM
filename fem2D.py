import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
import time



# right-hand side
f = -1.0

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
    def __init__(self, x, y, idx=-1):
        self.x, self.y = x, y
        self.idx = idx


class Grid:
    def __init__(self, xMin=0.0, xMax=1.0, yMin=0.0, yMax=1.0, xSteps=11, ySteps=11):
        self.xMin, self.xMax = xMin, xMax
        self.yMin, self.yMax = yMin, yMax
        self.hx = (xMax-xMin)/(xSteps-1)
        self.hy = (yMax-yMin)/(ySteps-1)
        self.xNum = xSteps
        self.yNum = ySteps
        self.Nodes, self.Elements = [], []

    def tictoc(func):
        def wrapper(self, *args, **kwargs):
            t1 = time.time()
            result = func(self)
            t2 = time.time() - t1
            print(f'{func.__name__} ran in {t2} seconds')
            return result
        return wrapper

    @tictoc
    def createMesh(self):
        # create mesh and transform it into a list of x and y coordinates
        xRange = np.linspace(self.xMin, self.xMax, self.xNum)
        yRange = np.linspace(self.yMin, self.yMax, self.yNum)
        self.xCoord, self.yCoord = np.meshgrid(xRange, yRange)
        xList, yList = self.xCoord.ravel(), self.yCoord.ravel()

        # create Nodes; Each node holds a nodal basis function
        for i, (x, y) in enumerate(zip(xList, yList)):
            self.Nodes.append(Node(x, y, idx=i))

        # create Elements
        for i, node in enumerate(self.Nodes):
            if node.x != self.xMax and node.y != self.yMax:
                self.Elements.append(
                    Element(
                        node,
                        self.Nodes[i + 1],
                        self.Nodes[i + self.xNum],
                        self.Nodes[i + self.xNum + 1],
                    )
                )

    @tictoc
    def assembleSystem(self):
        # system matrix
        A = dok_matrix((len(self.Nodes), len(self.Nodes)), dtype=np.float32)
        # system right hand side
        F = np.zeros(len(self.Nodes), dtype=np.float32)

        tx = time.time()
        for element in self.Elements:
            for x, y, quadWeight in [ (x, y, wX * wY) for (x, wX) in quad for (y, wY) in quad ]:
                for i, node_1 in enumerate(element.nodes):
                    # assemble rhs
                    F[node_1.idx] += (
                        quadWeight * f * Nodal2D[i].v_phi(x, y) * self.hx * self.hy
                    )
                    for j, node_2 in enumerate(element.nodes):
                        # assemble matrix
                        A[node_1.idx, node_2.idx] += quadWeight * Nodal2D[i].d_phi(x, y).dot(
                            Nodal2D[j].d_phi(x, y)
                        )
        ty = time.time()
        print(f"Create A and F: {ty-tx}s")
        # apply homogeneous Dirichlet boundary conditions
        for node in self.Nodes:
            if node.x in [self.xMin, self.xMax] or node.y in [self.yMin, self.yMax]:
                _, nonZeroColumns = A[node.idx, :].nonzero()
                for i in nonZeroColumns:
                    A[node.idx, i] = 0.0
                #set U at Dirichlet boundary = assigned value 
                A[node.idx, node.idx] = 1.0
                F[node.idx] = 0.0
        tz = time.time()
        print(f"Apply boundary condition: {tz-ty}s")
        return A.tocsr(), F

    def plotSolution(self, solution):
        # 2D contour plot
        plt.imshow(
            solution.reshape(self.yNum, self.xNum), extent = [self.xMin, self.xMax, self.yMin, self.yMax],
            cmap="jet",
        )
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    grid = Grid(xSteps=101, ySteps=101)
    grid.createMesh()
    A, F = grid.assembleSystem()

    t1 = time.time()
    solution = spsolve(A, F)
    #solution, exit_code = cg(A, F) # U = A^{-1}F
    t2 = time.time() - t1
    print(f"Sparse matrix solve time: {t2} seconds")
    grid.plotSolution(solution)
    
