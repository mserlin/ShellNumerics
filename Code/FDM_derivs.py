import numpy as np
import vandermonde #Compute vandermonde matrix 

from scipy.signal import convolve, convolve2d 
 
def genFDMCoefs(deriv, points):
    #deriv is order of the derivative
    #points is a list, with 0 being the derivative of interest.
    #for example,deriv 1 with points -1, 0, 1 should return -1/2 0 1/2
    
    rhs = np.zeros(np.shape(points))
    rhs[deriv] = np.math.factorial(deriv)
    
    return vandermonde.solve_transpose(points, rhs)
        
def reflectStencils(i, j, si, sj, Lpnts, rpnts, stencils):
        
        nmax = stencils.shape[0]
        
        #Modify the stencils if close to a symmetry point
        #Reflection about the x = L/2 axis
        if i + si[-1] > Lpnts - 1: 
            #index = np.where(j + sj == Lpnts - 1)[0][0]
            
            index = (si.size - 1) - (i + si[-1] - (Lpnts -1))
            
            #for k in range(si.size):
            #    if i + si[-1 - i] == Lpnts - 1:
            #        index = i 
            #        break
                    
            for k in range(si.size - 1 - index):
                for m in range(sj.size):
                    for n in range(nmax):
                        stencils[n,index - (k+1),m] = stencils[n,index - (k+1),m] + stencils[n,index + k + 1,m]
                        stencils[n,index + k + 1,m] = 0
        
        #Reflection about the y = pi*r axis
        if j + sj[-1] > rpnts - 1:
            #index = np.where(j + sj == rpnts - 1)[0][0]
            index = (sj.size - 1) - (j + sj[-1] - (rpnts -1))
            
            for k in range(sj.size - 1 - index):
                for m in range(si.size):
                    for n in range(nmax):
                        stencils[n, m, index - (k+1)] = stencils[n, m, index - (k+1)] + stencils[n, m,index + k + 1]
                        stencils[n, m, index + (k+1)] = 0
        
        #Reflection about the y = 0 axis
        elif j + sj[0] < 0:
            #index = np.where(j + sj == 0)[0][0] #can probably do this without np.where
            index = -(j + sj[0])
            for k in range(index):
                for m in range(si.size):
                    for n in range(nmax):
                        stencils[n, m, index + (k+1)] = stencils[n, m, index + (k+1)] + stencils[n, m, index - (k+1)]
                        stencils[n, m, index - (k+1)] = 0
        
        return stencils
    
class derivGenerator:
    def __init__(self, order, dx, dy, xpnts):
        self.order, self.dx, self.dy, self.xpnts = order, dx, dy, xpnts
        
        #Generate stencils for up to 4th order derivatives
        #First generate the stencils in the bulk, where we always have enough points on both sides of the central point 
        #to do central difference methods
        stencils_1d_bulk = np.zeros((5, 3+2*order))

        stencils_1d_bulk[0][order + 1] = 1
        stencils_1d_bulk[1] = np.pad(genFDMCoefs(1, np.arange(1+2*order) - order), 1)
        stencils_1d_bulk[2] = np.pad(genFDMCoefs(2, np.arange(1+2*order) - order), 1)
        stencils_1d_bulk[3] = genFDMCoefs(3, np.arange(3+2*order) - order - 1)
        stencils_1d_bulk[4] = genFDMCoefs(4, np.arange(3+2*order) - order - 1)
        
        stencils_2d_bulk = np.zeros((5, 5, 3+2*order, 3+2*order))
        for i in range(0,5):
            for j in range(0,5):
                stencils_2d_bulk[i,j] = np.outer(stencils_1d_bulk[i], stencils_1d_bulk[j])/(dx**i * dy**j)
        
        self.stencils_1d_bulk = stencils_1d_bulk
        self.stencils_2d_bulk = stencils_2d_bulk
        
        #Now lets generalize to also make stencils on the edge, where central difference cannot be made. 
        #Note, that since we have PBC in y, the y axis is always in the bulk. 
        #Adding a single point makes the order of the approximation the same
        
        stencils_1d_edge = np.zeros((order + 1, 5, 4+2*order))
        
        for k in range(order + 1):
            stencils_1d_edge[k, 0][k] = 1
            stencils_1d_edge[k, 1] = np.pad(genFDMCoefs(1, np.arange(1+2*order) - k), (0,3))
            stencils_1d_edge[k, 2] = np.pad(genFDMCoefs(2, np.arange(2+2*order) - k), (0,2))
            stencils_1d_edge[k, 3] = np.pad(genFDMCoefs(3, np.arange(3+2*order) - k), (0,1))
            stencils_1d_edge[k, 4] = genFDMCoefs(4, np.arange(4+2*order) - k)
        
        stencils_2d_edge = np.zeros((order+1, 5, 5, 4+2*order, 3+2*order))
        for k in range(order+1):
            for i in range(0,5):
                for j in range(0,5):
                    stencils_2d_edge[k,i,j] = np.outer(stencils_1d_edge[k, i], stencils_1d_bulk[j])/(dx**i * dy**j)

        self.stencils_1d_edge = stencils_1d_edge
        self.stencils_2d_edge = stencils_2d_edge
        
    def gen_stencil(self, x_deriv, y_deriv, i = None):
        if i is None or i > self.order and i < self.xpnts - 1 - self.order:
            return self.stencils_2d_bulk[x_deriv, y_deriv]
        elif i <= self.order:
            return self.stencils_2d_edge[i, x_deriv, y_deriv]
        elif i >= self.xpnts - 1 - self.order:
            return self.stencils_2d_edge[self.xpnts -1 - i, x_deriv, y_deriv][::-1, :]*((-1)**(x_deriv))
        
    def x_deriv(self, M, nx):
        #Takes an x derivative accounting for edge effects
        #Factor of -1 come form different sign convention from convolve2d 
        if nx > 0:
            coefs = (-1)**nx * np.reshape(self.stencils_1d_bulk[nx], (self.stencils_1d_bulk[nx].size, 1))
            Mx = convolve2d(M, coefs, 'same')/self.dx**nx
            
            for i in range(self.order+int((nx-1)/2)):
                coefs = np.reshape(self.stencils_1d_edge[i, nx], (1, self.stencils_1d_edge[i, nx].size))[:,:nx+2*self.order]
                Mx[i, :] = np.dot(coefs, M[:nx+2*self.order, :])/self.dx**nx
                Mx[-(i+1), :] = np.dot(coefs[:, ::-1]*(-1)**nx, M[-nx-2*self.order:, :])/self.dx**nx
            
            return Mx
        else:
            return M
        
    def y_deriv(self, M, ny):
        #Takes a y derivative accounting for PBCs 
        if ny > 0:
            coefs = (-1)**ny*np.reshape(self.stencils_1d_bulk[ny], (1, self.stencils_1d_bulk[ny].size))
            My = convolve2d(M, coefs, 'same')/self.dy**ny
            
            for i in range(self.order+1): #int((ny-1)/2)
                My[:, i] = 0
                My[:, -(i+1)] = 0
                
                hlf = int((ny-1)/2)
                pnts = 1 + 2*self.order + 2*hlf
                offset = self.order + hlf
                for j in range(0, pnts):
                    My[:, i] += self.stencils_1d_bulk[ny][j + 1 - hlf]*M[:, i + j - offset]/self.dy**ny
                    My[:, -(i+1)] += self.stencils_1d_bulk[ny][j + 1 - hlf]*M[:, -(i+1) + j - offset]/self.dy**ny
            return My
        else:
            return M
            
    def deriv(self, M, nx, ny):
        return self.y_deriv(self.x_deriv(M, nx), ny)