import numpy as np
from scipy.sparse import csc_matrix
import time 

class fastJacobian:
    def __init__(self, r, L, nu, dgen, w0_derivs, t_derivs):
        self.r, self.L, self.nu, = r, L, nu
        self.dgen = dgen
        self.w0_derivs = w0_derivs
        self.t_derivs = t_derivs
        
    def calcJacob(self, double[:,:] w, double[:,:] u, double[:,:] v, bc_type = 'disp'):
        
        t0 = time.time()
        
        cdef double r = self.r
        cdef double L = self.L
        cdef double nu = self.nu
        
        cdef double[:] consts = np.array([nu, r])
        
        cdef int Lpnts = w.shape[0]
        cdef int rpnts = w.shape[1]
        
        dgen = self.dgen
        cdef int order = dgen.order
        
        #Calculate the stencils for all the required derivatives in the bulk 
        #Edge cases are calculated in the loop
        
        bulk_stencils_np = np.zeros((13, 3+2*order, 3+2*order))
        bulk_stencils_np[0,:,:] = dgen.gen_stencil(0,0)
        
        bulk_stencils_np[1,:,:] = dgen.gen_stencil(1,0)
        bulk_stencils_np[2,:,:] = dgen.gen_stencil(0,1)
        
        bulk_stencils_np[3,:,:] = dgen.gen_stencil(2,0)
        bulk_stencils_np[4,:,:] = dgen.gen_stencil(1,1)
        bulk_stencils_np[5,:,:] = dgen.gen_stencil(0,2)
        
        bulk_stencils_np[6,:,:] = dgen.gen_stencil(3,0)
        bulk_stencils_np[7,:,:] = dgen.gen_stencil(2,1)
        bulk_stencils_np[8,:,:] = dgen.gen_stencil(1,2)
        bulk_stencils_np[9,:,:] = dgen.gen_stencil(0,3)
        
        bulk_stencils_np[10,:,:] = dgen.gen_stencil(4,0)
        bulk_stencils_np[11,:,:] = dgen.gen_stencil(2,2)
        bulk_stencils_np[12,:,:] = dgen.gen_stencil(0,4)
        
        cdef double[:,:,:] bulk_stencils = bulk_stencils_np

        edge_stencils_np_low = np.zeros((order+1, 13, 4+2*order, 3+2*order))
        edge_stencils_np_low[:,0,:,:] = dgen.stencils_2d_edge[:,0,0]
        
        edge_stencils_np_low[:,1,:,:] = dgen.stencils_2d_edge[:,1,0]
        edge_stencils_np_low[:,2,:,:] = dgen.stencils_2d_edge[:,0,1]
        
        edge_stencils_np_low[:,3,:,:] = dgen.stencils_2d_edge[:,2,0]
        edge_stencils_np_low[:,4,:,:] = dgen.stencils_2d_edge[:,1,1]
        edge_stencils_np_low[:,5,:,:] = dgen.stencils_2d_edge[:,0,2]
        
        edge_stencils_np_low[:,6,:,:] = dgen.stencils_2d_edge[:,3,0]
        edge_stencils_np_low[:,7,:,:] = dgen.stencils_2d_edge[:,2,1]
        edge_stencils_np_low[:,8,:,:] = dgen.stencils_2d_edge[:,1,2]
        edge_stencils_np_low[:,9,:,:] = dgen.stencils_2d_edge[:,0,3]
        
        edge_stencils_np_low[:,10,:,:] = dgen.stencils_2d_edge[:,4,0]
        edge_stencils_np_low[:,11,:,:] = dgen.stencils_2d_edge[:,2,2]
        edge_stencils_np_low[:,12,:,:] = dgen.stencils_2d_edge[:,0,4]

        cdef double[:,:,:,:] edge_stencils_low = edge_stencils_np_low
        
        edge_stencils_np_high = edge_stencils_np_low.copy()[:,:,::-1,:]
        edge_stencils_np_high[:,1,:,:] = edge_stencils_np_high[:,1,:,:]*-1
        edge_stencils_np_high[:,4,:,:] = edge_stencils_np_high[:,4,:,:]*-1
        edge_stencils_np_high[:,6,:,:] = edge_stencils_np_high[:,6,:,:]*-1
        edge_stencils_np_high[:,8,:,:] = edge_stencils_np_high[:,8,:,:]*-1

        cdef double[:,:,:,:] edge_stencils_high = edge_stencils_np_high
        
        cdef int i_bc0 = 1
        cdef int i_bcL = Lpnts - 2
        
        cdef int[:] sbi = np.arange(3+2*order) - order - 1
        cdef int[:] sj = np.arange(3+2*order) - order - 1
        
        #Stresses are inputs to the formulas at places, so calculate those first
        #Get the w, w0, u, v deriv arrays
        w_derivs_np = np.zeros((6, Lpnts, rpnts))
        w_derivs_np[0,:,:] = w
        w_derivs_np[1,:,:] = dgen.deriv(np.array(w), 1, 0)
        w_derivs_np[2,:,:] = dgen.deriv(np.array(w), 0, 1)
        w_derivs_np[3,:,:] = dgen.deriv(np.array(w), 2, 0)
        w_derivs_np[4,:,:] = dgen.deriv(np.array(w), 1, 1)
        w_derivs_np[5,:,:] = dgen.deriv(np.array(w), 0, 2)
        cdef double[:,:,:] w_derivs = w_derivs_np
        
        u_derivs_np = np.zeros((5, Lpnts, rpnts))
        u_derivs_np[0,:,:] = dgen.deriv(np.array(u), 1, 0)
        u_derivs_np[1,:,:] = dgen.deriv(np.array(u), 0, 1)
        u_derivs_np[2,:,:] = dgen.deriv(np.array(u), 2, 0)
        u_derivs_np[3,:,:] = dgen.deriv(np.array(u), 1, 1)
        u_derivs_np[4,:,:] = dgen.deriv(np.array(u), 0, 2)
        cdef double[:,:,:] u_derivs = u_derivs_np

        v_derivs_np = np.zeros((5, Lpnts, rpnts))
        v_derivs_np[0,:,:] = dgen.deriv(np.array(v), 1, 0)
        v_derivs_np[1,:,:] = dgen.deriv(np.array(v), 0, 1)
        v_derivs_np[2,:,:] = dgen.deriv(np.array(v), 2, 0)
        v_derivs_np[3,:,:] = dgen.deriv(np.array(v), 1, 1)
        v_derivs_np[4,:,:] = dgen.deriv(np.array(v), 0, 2)
        cdef double[:,:,:] v_derivs = v_derivs_np
        
        w0_derivs_np = self.w0_derivs
        cdef double[:,:,:] w0_derivs = w0_derivs_np
        cdef double[:,:,:] t_derivs = self.t_derivs
        
        ep1 = u_derivs_np[0] + w_derivs_np[1]**2/2 + w_derivs_np[1]*w0_derivs_np[1]
        ep2 = v_derivs_np[1] - w_derivs_np[0]/r + w_derivs_np[2]**2/2 + w_derivs_np[2]*w0_derivs_np[2]
        gamma = v_derivs_np[0] + u_derivs_np[1] + w_derivs_np[1]*w_derivs_np[2] + w0_derivs_np[1]*w_derivs_np[2] + w_derivs_np[1]*w0_derivs_np[2]
        
        ep1_y = u_derivs_np[3] + w_derivs_np[4]*(w_derivs_np[1] + w0_derivs_np[1]) + w_derivs_np[1]*w0_derivs_np[4]
        ep2_y = v_derivs_np[4] - (w_derivs_np[2])*(1/r - w_derivs_np[5] - w0_derivs_np[5]) + w_derivs_np[5]*w0_derivs_np[2] 
                
        cdef double[:,:,:] sigx_derivs = np.reshape((ep1 + nu*ep2)/(1-nu**2), (1, Lpnts, rpnts))
        cdef double[:,:,:] sigy_derivs = np.reshape([(nu*ep1 + ep2)/(1-nu**2), (nu*ep1_y + ep2_y)/(1-nu**2)], (2, Lpnts, rpnts))
        cdef double[:,:,:] tau_derivs = np.reshape(gamma/(2*(1+nu)), (1, Lpnts, rpnts))
        
        #Initialize variables that will be assigned values in the loop
        cdef int max_points = (4+2*order)*(4+2*order)*w.size*3*3
        
        cdef int[:] data_ind = np.zeros(1, dtype = 'int')
        cdef double[:] matrixData = np.zeros(max_points, dtype = 'float')
        cdef int[:] matrixRow = np.zeros(max_points, dtype = 'int')
        cdef int[:] matrixCol = np.zeros(max_points, dtype = 'int')
        
        cdef int i, i_calc, j, k, m, n, ind_1D_w, ind_1D_u, ind_1D_v
        cdef double[:,:] s_0, s_dx
        
        cdef int[:] si
        cdef double[:,:,:] stencils
        cdef double[:] partials = np.zeros(9)
        
        for i in range(Lpnts):
            #First get stencils for the current axial position (it varies when close to the edge)
            #If in bulk use bulk stencils
            if i > order and i < Lpnts - 1 - order:
                si = sbi
                stencils = bulk_stencils
                #s_0, s_dx, s_dy, s_dxx, s_dxy, s_dyy, s_dxxxx, s_dxxyy, s_dyyyy = sb_0, sb_dx, sb_dy, sb_dxx, sb_dxy, sb_dyy, sb_dxxxx, sb_dxxyy, sb_dyyyy
            #Otherwise generate edge stencils 
            else:
                #Boundary conditions for i = 0 and i = Lpnts-1 are calculated for the second point from the end
                if i == 0:
                    i_calc = i_bc0
                elif i == Lpnts-1:
                    i_calc = i_bcL
                else:
                    i_calc = i
                
                if i_calc <= order:
                    stencils = edge_stencils_low[i_calc]
                    si = np.arange(4+2*order) - i_calc
                else:
                    stencils = edge_stencils_high[Lpnts -1 - i_calc]
                    si = -np.arange(4+2*order)[::-1] + (Lpnts - 1 - i_calc)
            
            for j in range(rpnts):
                #i corresponds to x, where x(i = 0) = -order_x * hx, x(i = order_x) = 0, x(i = Lpnts -1 + order_x) = L, x(i = Lpnts -1 + 2*order_x) = L + order_x*hx
                #j corresponds to y, where y(j = 0) = 0 and y(j = rpnts - 1) = 2pi r - hy. 
                
                ind_1D_w = calc1DIndex(i,j, 0, Lpnts, rpnts)
                ind_1D_u = calc1DIndex(i,j, 1, Lpnts, rpnts)
                ind_1D_v = calc1DIndex(i,j, 2, Lpnts, rpnts)
                
                #first take care of boundary conditions
                if i == 0:
                    #w(x=0) = 0 or nu*r*load
                    setValues(data_ind, matrixData, matrixRow, matrixCol, 1, ind_1D_w, calc1DIndex(i_bc0, j, 0, Lpnts, rpnts))
                    
                    #u(x=0) = 0
                    setValues(data_ind, matrixData, matrixRow, matrixCol, 1, ind_1D_u, calc1DIndex(i_bc0, j, 1, Lpnts, rpnts))
                    
                    #v(x=0) = 0
                    setValues(data_ind, matrixData, matrixRow, matrixCol, 1, ind_1D_v, calc1DIndex(i_bc0, j, 2, Lpnts, rpnts))
                    
                elif i == 1:
                    #wx(x=L) = 0
                    for m in range(si.size):
                        i_ind = i_bc0 + si[m]
                        for n in range(sj.size):
                            j_ind = j+sj[n]
                            if stencils[1][m,n] != 0:
                                setValues(data_ind, matrixData, matrixRow, matrixCol, stencils[1][m,n], ind_1D_w, calc1DIndex(i_ind, j_ind, 0, Lpnts, rpnts))
                                
                            #add the regular u and v stability equations
                            
                            #Calculate the partials for the current values of i,j,m,n 
                            calcPartials(partials, w_derivs[:,i,j], u_derivs[:,i,j], v_derivs[:,i,j], w0_derivs[:,i,j], t_derivs[:,i,j], sigx_derivs[:,i,j], sigy_derivs[:,i,j], tau_derivs[:,i,j], stencils[:,m,n], consts)
                            
                            for k in range(3):
                                setValues(data_ind, matrixData, matrixRow, matrixCol, partials[3+k], ind_1D_u, calc1DIndex(i_ind, j_ind, k, Lpnts, rpnts))
                                setValues(data_ind, matrixData, matrixRow, matrixCol, partials[6+k], ind_1D_v, calc1DIndex(i_ind, j_ind, k, Lpnts, rpnts))
                            
                elif i == Lpnts - 2:
                    #wx(x=L) = 0
                    for m in range(si.size):
                        i_ind = i_bcL + si[m]
                        for n in range(sj.size):
                            j_ind = j+sj[n]
                            if stencils[1][m,n] != 0:
                                setValues(data_ind, matrixData, matrixRow, matrixCol, stencils[1][m,n], ind_1D_w, calc1DIndex(i_ind,j_ind, 0, Lpnts, rpnts))
                                
                            #add the regular u and v stability equations
                            
                            #Calculate the partials for the current values of i,j,m,n 
                            calcPartials(partials, w_derivs[:,i,j], u_derivs[:,i,j], v_derivs[:,i,j], w0_derivs[:,i,j], t_derivs[:,i,j], sigx_derivs[:,i,j], sigy_derivs[:,i,j], tau_derivs[:,i,j], stencils[:,m,n], consts)
                            
                            #Add the partials to the Jacobian
                            for k in range(3):
                                setValues(data_ind, matrixData, matrixRow, matrixCol, partials[3+k], ind_1D_u, calc1DIndex(i_ind, j_ind, k, Lpnts, rpnts))
                                setValues(data_ind, matrixData, matrixRow, matrixCol, partials[6+k], ind_1D_v, calc1DIndex(i_ind, j_ind, k, Lpnts, rpnts))
                            
                elif i == Lpnts - 1:
                    #w(x=L) = 0
                    setValues(data_ind, matrixData, matrixRow, matrixCol, 1, ind_1D_w, calc1DIndex(i_bcL, j, 0, Lpnts, rpnts))
                    
                    #v(x = L) = 0
                    setValues(data_ind, matrixData, matrixRow, matrixCol, 1, ind_1D_v, calc1DIndex(i_bcL, j, 2, Lpnts, rpnts))
                    
                    if bc_type == 'disp':
                        #u(x = L) = whatever it needs to be
                        setValues(data_ind, matrixData, matrixRow, matrixCol, 1, ind_1D_u, calc1DIndex(i_bcL, j, 1, Lpnts, rpnts))
                    elif bc_type == 'load':
                        if j == 0:
                            #Average applied load
                            #sigx = (ep1 + nu ep2)/(1-nu**2)
                            #ep1 = ux + wx^2/2 + wx w0x
                            #ep2 = vy - w/r + wy^2/2 + wy w0y
                            #vy, wy, wx are zero. So, simplifies to
                            #sigx = (ux - nu w / r)/(1-nu^2)
                            s_0 = stencils[0]
                            s_dx = stencils[1]
                            
                            for m in range(si.size):
                                i_ind = i_bcL + si[m]
                                for n in range(sj.size):
                                    j_ind = sj[n]
                                    
                                    sigx_w = (nu*s_0[m,n])/(1-nu**2)/rpnts
                                    sigx_u = (s_dx[m,n])/(1-nu**2)/rpnts
                                    
                                    if sigx_u != 0:
                                        for k in range(rpnts):
                                            setValues(data_ind, matrixData, matrixRow, matrixCol, sigx_u, ind_1D_u, calc1DIndex(i_ind, k+j_ind, 1, Lpnts, rpnts))
                                    
                                    if sigx_w != 0:
                                        for k in range(rpnts):
                                            setValues(data_ind, matrixData, matrixRow, matrixCol, sigx_w, ind_1D_u, calc1DIndex(i_ind, k+j_ind, 0, Lpnts, rpnts))
                        else:
                            #u(x=L, y) - u(x=L, y+dy) = 0
                            setValues(data_ind, matrixData, matrixRow, matrixCol, 1, ind_1D_u, calc1DIndex(i_bcL, j-1, 1, Lpnts, rpnts))
                            setValues(data_ind, matrixData, matrixRow, matrixCol, -1, ind_1D_u, calc1DIndex(i_bcL, j, 1, Lpnts, rpnts))
                    
                else:
                    for m in range(si.size):
                        i_ind = i+si[m]
                        for n in range(sj.size):
                            j_ind = j+sj[n]
                            
                            #Calculate the partials for the current values of i,j,m,n 
                            calcPartials(partials, w_derivs[:,i,j], u_derivs[:,i,j], v_derivs[:,i,j], w0_derivs[:,i,j], t_derivs[:,i,j], sigx_derivs[:,i,j], sigy_derivs[:,i,j], tau_derivs[:,i,j], stencils[:,m,n], consts)
                            
                            #Add the partials to the Jacobian
                            for k in range(3):
                                setValues(data_ind, matrixData, matrixRow, matrixCol, partials[k], ind_1D_w, calc1DIndex(i_ind, j_ind, k, Lpnts, rpnts))
                                setValues(data_ind, matrixData, matrixRow, matrixCol, partials[3+k], ind_1D_u, calc1DIndex(i_ind, j_ind, k, Lpnts, rpnts))
                                setValues(data_ind, matrixData, matrixRow, matrixCol, partials[6+k], ind_1D_v, calc1DIndex(i_ind, j_ind, k, Lpnts, rpnts))
                            
        matrixData = matrixData[0:data_ind[0]]
        matrixRow = matrixRow[0:data_ind[0]]
        matrixCol = matrixCol[0:data_ind[0]]
        
        matrix = (matrixData, (matrixRow, matrixCol))
        
        jacobian = csc_matrix(matrix, shape = (3*Lpnts*rpnts, 3*Lpnts*rpnts))
        jacobian.sort_indices() #Sort indices for pypardiso compatibility
        
        return jacobian


cdef void calcPartials(double[:] partials, double[:] w_derivs, double[:] u_derivs, double[:] v_derivs, double[:] w0_derivs, double[:] t_derivs, double[:] sigx_derivs, double[:] sigy_derivs, double[:] tau_derivs, double[:] stencils, double[:] consts):
        
    cdef double nu = consts[0]
    cdef double r = consts[1]
    
    #Initialize deriv arrays
    cdef double w = w_derivs[0]
    cdef double wx = w_derivs[1]
    cdef double wy = w_derivs[2]
    cdef double wxx = w_derivs[3]
    cdef double wxy = w_derivs[4]
    cdef double wyy = w_derivs[5]
    
    cdef double ux = u_derivs[0]
    cdef double uy = u_derivs[1]
    cdef double uxx = u_derivs[2]
    cdef double uxy = u_derivs[3]
    cdef double uyy = u_derivs[4]
    
    cdef double vx = v_derivs[0]
    cdef double vy = v_derivs[1]
    cdef double vxx = v_derivs[2]
    cdef double vxy = v_derivs[3]
    cdef double vyy = v_derivs[4]
    
    cdef double w0 = w0_derivs[0]
    cdef double w0x = w0_derivs[1]
    cdef double w0y = w0_derivs[2]
    cdef double w0xx = w0_derivs[3]
    cdef double w0xy = w0_derivs[4]
    cdef double w0yy = w0_derivs[5]
    
    cdef double t = t_derivs[0]
    cdef double tx = t_derivs[1]
    cdef double ty = t_derivs[2]
    cdef double txx = t_derivs[3]
    cdef double txy = t_derivs[4]
    cdef double tyy = t_derivs[5]
    
    cdef double s_0 = stencils[0]
    cdef double s_dx = stencils[1]
    cdef double s_dy = stencils[2]
    cdef double s_dxx = stencils[3]
    cdef double s_dxy = stencils[4]
    cdef double s_dyy = stencils[5]
    cdef double s_dxxx = stencils[6]
    cdef double s_dxxy = stencils[7]
    cdef double s_dxyy = stencils[8]
    cdef double s_dyyy = stencils[9]
    cdef double s_dxxxx = stencils[10]
    cdef double s_dxxyy = stencils[11]
    cdef double s_dyyyy = stencils[12]
    
    cdef double sigx = sigx_derivs[0]
    
    cdef double sigy = sigy_derivs[0]
    cdef double sigy_y = sigy_derivs[1]
    
    cdef double tau = tau_derivs[0]
    
    cdef double sigx_w, sigx_u, sigx_v, sigx_x_w, sigx_x_u, sigx_x_v
    cdef double sigy_w, sigy_u, sigy_v, sigy_y_w, sigy_y_u, sigy_y_v
    cdef double tau_w, tau_u, tau_v, tau_x_w, tau_x_u, tau_x_v, tau_y_w, tau_y_u, tau_y_v
        
    cdef double ep1_w, ep1_u, ep1_v, ep1_x_w, ep1_x_u, ep1_x_v, ep1_y_w, ep1_y_u, ep1_y_v
    cdef double ep2_w, ep2_u, ep2_v, ep2_x_w, ep2_x_u, ep2_x_v, ep2_y_w, ep2_y_u, ep2_y_v
    cdef double gamma_w, gamma_u, gamma_v, gamma_x_w, gamma_x_u, gamma_x_v, gamma_y_w, gamma_y_u, gamma_y_v
    
    cdef double w_s_w, w_s_u, w_s_v
    cdef double u_s_w, u_s_u, u_s_v
    cdef double v_s_w, v_s_u, v_s_v
    
    #First, lets calculate some commonly used partials. 
    #Lets start with the strains

    #ep1 = ux + wx**2/2 + wx*w0x
    ep1_w = s_dx*(wx + w0x)
    ep1_u = s_dx
    ep1_v = 0
    
    ep1_x_w = s_dxx*(wx + w0x) + s_dx*(wxx + w0xx)
    ep1_x_u = s_dxx
    ep1_x_v = 0
    
    ep1_y_w = s_dxy*(wx + w0x) + s_dx*(wxy + w0xy)
    ep1_y_u = s_dxy
    ep1_y_v = 0
    
    #ep2 = vy - w/r + wy**2/2 + wy*w0y
    ep2_w = -s_0/r + s_dy*(wy + w0y)
    ep2_u = 0
    ep2_v = s_dy
    
    ep2_x_w = -s_dx/r + s_dxy*(wy + w0y) + s_dy*(wxy + w0xy)
    ep2_x_u = 0
    ep2_x_v = s_dxy
    
    ep2_y_w = -s_dy/r + s_dyy*(wy + w0y) + s_dy*(wyy + w0yy)
    ep2_y_u = 0
    ep2_y_v = s_dyy
    
    #gamma = vx + uy + wx*wy + w0x*wy + wx*w0y
    gamma_w = s_dx*(wy+w0y) + s_dy*(wx+w0x)
    gamma_u = s_dy
    gamma_v = s_dx
    
    gamma_x_w = s_dxx*(wy+w0y)+s_dx*(wxy+w0xy) + s_dxy*(wx+w0x)+ s_dy*(wxx+w0xx)
    gamma_x_u = s_dxy
    gamma_x_v = s_dxx
    
    gamma_y_w = s_dxy*(wy+w0y)+s_dx*(wyy+w0yy) + s_dyy*(wx+w0x)+s_dy*(wxy+w0xy)
    gamma_y_u = s_dyy
    gamma_y_v = s_dxy
    
    #Notation: sigx_y_w is the w partial of the y partial of the axial stress (sigx)
    sigx_w = (ep1_w + nu*ep2_w)/(1-nu**2)
    sigx_u = (ep1_u + nu*ep2_u)/(1-nu**2)
    sigx_v = (ep1_v + nu*ep2_v)/(1-nu**2)
    
    sigx_x_w = (ep1_x_w + nu*ep2_x_w)/(1-nu**2)
    sigx_x_u = (ep1_x_u + nu*ep2_x_u)/(1-nu**2)
    sigx_x_v = (ep1_x_v + nu*ep2_x_v)/(1-nu**2)
    
    sigy_w = (nu*ep1_w + ep2_w)/(1-nu**2)
    sigy_u = (nu*ep1_u + ep2_u)/(1-nu**2)
    sigy_v = (nu*ep1_v + ep2_v)/(1-nu**2)
    
    sigy_y_w = (nu*ep1_y_w + ep2_y_w)/(1-nu**2)
    sigy_y_u = (nu*ep1_y_u + ep2_y_u)/(1-nu**2)
    sigy_y_v = (nu*ep1_y_v + ep2_y_v)/(1-nu**2)
    
    tau_w = gamma_w/(2+2*nu)
    tau_u = gamma_u/(2+2*nu)
    tau_v = gamma_v/(2+2*nu)
    
    tau_x_w = gamma_x_w/(2+2*nu)
    tau_x_u = gamma_x_u/(2+2*nu)
    tau_x_v = gamma_x_v/(2+2*nu)
    
    tau_y_w = gamma_y_w/(2+2*nu)
    tau_y_u = gamma_y_u/(2+2*nu)
    tau_y_v = gamma_y_v/(2+2*nu)
    
    w_s_w = (t**3/(12*(1-nu**2)))*(s_dxxxx + 2*s_dxxyy + s_dyyyy)
    
    w_s_w += t**2 * txx *(nu*s_dyy + s_dxx) / (4*(1-nu**2))
    w_s_w += t**2 * tyy *(s_dyy + nu*s_dxx) / (4*(1-nu**2))
    w_s_w += t**2 * txy *(s_dxy) / (2*(1+nu))
    
    w_s_w += t * tx * (tx *(nu*s_dyy + s_dxx) + t*(s_dxyy + s_dxxx)) / (2*(1-nu**2))
    w_s_w += t * ty * (ty *(s_dyy + nu*s_dxx) + t*(s_dyyy + s_dxxy)) / (2*(1-nu**2))
    
    #Next sigx (wxx + w0xx) term
    w_s_w -= t*(wxx + w0xx)*sigx_w
    w_s_w -= t*s_dxx*sigx
    
    #Next sigy(1/r + wyy + w0yy) term
    w_s_w -= t*(1/r + wyy + w0yy)*sigy_w
    w_s_w -= t*s_dyy*sigy
    
    #Next tau(wxy  w0xy) term
    w_s_w -= 2*t*(wxy+w0xy)*tau_w
    w_s_w -= 2*t*s_dxy*tau
    
    #Repeat process for partial u
    w_s_u = -t*(1/r + wyy + w0yy)*sigy_u - t*(wxx + w0xx)*sigx_u - 2*t*(wxy + w0xy)*tau_u
    
    #Again for v
    w_s_v = -t*(1/r + wyy + w0yy)*sigy_v - t*(wxx + w0xx)*sigx_v - 2*t*(wxy + w0xy)*tau_v
    
    #u stability equation
    # t sigx_x + t_x sigx + t tau_y + ty tau= 0 
    u_s_w = t*sigx_x_w + tx*sigx_w + t*tau_y_w + ty*tau_w
    u_s_u = t*sigx_x_u + tx*sigx_u + t*tau_y_u + ty*tau_u
    u_s_v = t*sigx_x_v + tx*sigx_v + t*tau_y_v + ty*tau_v
    
    #v stability equation
    #t sigy_y + ty sigy + t tau_x + tx tau= 0
    v_s_w = t*sigy_y_w + ty*sigy_w + t*tau_x_w + tx*tau_w
    v_s_u = t*sigy_y_u + ty*sigy_u + t*tau_x_u + tx*tau_u
    v_s_v = t*sigy_y_v + ty*sigy_v + t*tau_x_v + tx*tau_v
    
    partials[0] = w_s_w
    partials[1] = w_s_u
    partials[2] = w_s_v
    
    partials[3] = u_s_w
    partials[4] = u_s_u
    partials[5] = u_s_v
    
    partials[6] = v_s_w
    partials[7] = v_s_u
    partials[8] = v_s_v

cdef double[:,:,:] reflectStencils(int i, int j, int[:] si, int[:] sj, int Lpnts, int rpnts, double [:,:,:] stencils):
        
        cdef int nmax = stencils.shape[0]
        cdef int k, m, n, index
        
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
                        stencils[n,index - (k+1),m] = stencils[n,index - (k+1),m] + stencils[n,index + (k+1),m]
                        stencils[n,index + (k+1),m] = 0
        
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

cdef int calc1DIndex(int i, int j, int disp, int Lpnts, int rpnts):
    #disp = 0 is w, 1 is u, 2 is v
    #calculates the 1D index from a 2D coordinates 
    cdef int ind
        
    if j >= 0 and j <= rpnts-1:
        ind = j + rpnts*i
    elif j < 0:
        ind = rpnts + j + rpnts*i
    elif j > rpnts-1:
        ind = j - rpnts + rpnts*i
        
    if disp == 0:
        return ind
    elif disp == 1:
        return ind + Lpnts*rpnts
    elif disp == 2:
        return ind + 2*Lpnts*rpnts

cdef void setValues(int[:] ind, double[:] matrixData, int[:] matrixRow, int[:] matrixCol, double val, int row, int col):
    if val != 0:
        matrixData[ind[0]] = val
        matrixRow[ind[0]] = row
        matrixCol[ind[0]] = col
        ind[0] += 1