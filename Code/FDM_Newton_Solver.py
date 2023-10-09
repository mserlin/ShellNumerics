import numpy as np
import time
import sys
from pathlib import Path
import matplotlib.pyplot as plot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import clear_output #For ensuring the IPython display doesn't get overrun with print statements
import pypardiso #Multicore sparse matrix solver
import pandas

#Import custom functions 
from FastJacobian import fastJacobian #Import function to rapidly create sparse jacobian matrices
from FDM_derivs import derivGenerator #Import discerete differentiation object

class FDM_Solver:
    def __init__(self, r, L, nu, w0_derivs, t_derivs, order = 2):
        #Define physical parameters of the cylinder to the simulated
        self.r = r
        self.L = L
        self.nu = nu
        
        #Number of points in the axial direction (Lpnts) and circumferential direction (rpnts)
        self.Lpnts = w0_derivs[1].shape[0]
        self.rpnts = w0_derivs[1].shape[1]
        
        #Define the uniform mesh on which it should be simulated
        self.dx = L/(self.Lpnts-1)
        self.dy = 2*np.pi*r / self.rpnts
        
        #Create the differentiation / stencil generating object based on the order of the derivative approximation
        #the mesh size, and the number of points in the x direction
        
        #The solver uses one ghost point is used on each side of the cylinder, so the total number of points in the axial direction is +2
        
        #Store differentiation order for the diffeq and the jacobian
        self.order = order 
        #Create the differentiation object
        self.dgen = derivGenerator(order, self.dx, self.dy, self.Lpnts+2)
        
        #Initialize grids accounting for one ghost point on either side
        self.X, self.Y = np.mgrid[-self.dx:L+self.dx:(self.Lpnts+2)*1j, 0:2*np.pi*r-self.dy:self.rpnts*1j]
        
        self.w0_derivs = np.pad(w0_derivs, ((0,0), (1,1), (0,0)))
        self.t_derivs = np.pad(t_derivs, ((0,0), (1,1), (0,0)), mode = 'mean')
        
        #Create Jacobian calculating object
        self.fastJacob = fastJacobian(self.r, self.L, self.nu, self.dgen, self.w0_derivs, self.t_derivs)
            
    
    def computeStrains(self, w, u, v):
        """
        Compute strains of the middle surface associated with deformations specified by w, u, and v. 

        Arguments
          w: deformations in the radial direction 
          u: deformations in the axial direction
          v: deformations in the circumferential direction
            
        Returns
          ep1: strain in the axial direction 
          ep2: strain in the circumferential direction
          gamma: shear strain 
        """
        w0, w0x, w0y, w0xx, w0xy, w0yy = self.w0_derivs
        deriv = self.dgen.deriv
        
        r = self.r
        
        #Now calc w derivatives
        wx = deriv(w, 1, 0)
        wy = deriv(w, 0, 1)
        
        #Calculate v derivatives
        vx = deriv(v, 1, 0)
        vy = deriv(v, 0, 1)
            
        #Calculate u derivatives
        ux = deriv(u, 1, 0)
        uy = deriv(u, 0, 1)
        
        #Calculate strains then stresses
        ep1 = ux + wx**2/2 + wx*w0x
        ep2 = vy - w/r + wy**2/2 + wy*w0y
        gamma = vx + uy + wx*wy + w0x*wy + wx*w0y

        return ep1, ep2, gamma
    
    def computeStresses(self, w, u, v):
        """
        Compute stresses in the middle surface associated with deformations specified by w, u, and v. 

        Arguments
          w: deformations in the radial direction 
          u: deformations in the axial direction
          v: deformations in the circumferential direction
            
        Returns
          sigx: stress in the axial direction 
          sigy: stress in the circumferential direction
          tau: shear stress
        """
        ep1, ep2, gamma = self.computeStrains(w, u, v)
        nu = self.nu
        
        sigx = (ep1 + nu*ep2)/(1-nu**2)
        sigy = (nu*ep1 + ep2)/(1-nu**2)
        tau = (gamma)/(2*(1+nu))
        
        return sigx, sigy, tau
        
    def calcErrorCombined(self, w, u, v, end_disp, bc_type = 'disp'):
        """
        The von-Karman Donnell equations are derived assuming that the total energy of a thin shell is a function of w, u, and v. 
        The functional derivative of the energy with respect to each of these variables defines three distinct partial differential
        equations that must simultaneously be satisfied in order for the shell to be in equilibrium. 
        This function calculate how far off the provided guess for w, u and v are from solving the equations.
        
        Arguments
          w: deformations in the radial direction 
          u: deformations in the axial direction
          v: deformations in the circumferential direction
          end_disp: End displacement or axial load on the shell
          bc_type: either 'disp' or 'load' for end displacement or axial load specified boundary conditions 
            
        Returns
          err_w: error in partial Energy partial w
          err_u: error in partial Energy partial u
          err_v: error in partial Energy partial v
        """
        #Re-define class values for code conciseness 
        deriv = self.dgen.deriv
        nu = self.nu
        r = self.r
        
        w0, w0x, w0y, w0xx, w0xy, w0yy = self.w0_derivs
        t, tx, ty, txx, txy, tyy = self.t_derivs
        
        D = t**3/(12*(1-nu**2))
        
        #Calc w derivatives
        wx, wy = deriv(w, 1, 0), deriv(w, 0, 1)
        wxx, wxy, wyy = deriv(w, 2, 0), deriv(w, 1, 1), deriv(w, 0, 2)
        wxxx, wxxy, wxyy, wyyy = deriv(w, 3, 0), deriv(w, 2, 1), deriv(w, 1, 2), deriv(w, 0, 3)
        wxxxx, wxxyy, wyyyy = deriv(w, 4, 0), deriv(w, 2, 2), deriv(w, 0, 4)
        
        #Calculate v derivatives
        vx, vy = deriv(v, 1, 0), deriv(v, 0, 1)
        vxx, vxy, vyy = deriv(v, 2, 0), deriv(v, 1, 1), deriv(v, 0, 2)
        
        #Calculate u derivatives
        ux, uy = deriv(u, 1, 0), deriv(u, 0, 1)
        uxx, uxy, uyy = deriv(u, 2, 0), deriv(u, 1, 1), deriv(u, 0, 2)
        
        ep1 = ux + wx**2/2 + wx*w0x
        ep2 = vy - w/r + wy**2/2 + wy*w0y
        gamma = vx + uy + wx*wy + w0x*wy + wx*w0y
        
        ep1_x = uxx + wxx*(wx + w0x) + wx*w0xx
        ep1_y = uxy + wxy*(wx + w0x) + wx*w0xy
        
        ep2_x = vxy - wx/r + wxy*(wy + w0y) + w0xy*wy
        ep2_y = vyy - wy/r + wyy*(wy + w0y) + w0yy*wy 
        
        gamma_x = vxx + uxy + wxy*(wx+w0x) + wxx*(wy+w0y) + w0xx*wy + wx*w0xy
        gamma_y = vxy + uyy + wxy*(wy+w0y) + wyy*(wx+w0x) + w0xy*wy + wx*w0yy

        sigx = (ep1 + nu*ep2)/(1-nu**2)
        sigy = (nu*ep1 + ep2)/(1-nu**2)
        tau = (gamma)/(2*(1+nu))
        
        sigx_x = (ep1_x + nu*ep2_x)/(1-nu**2)
        sigy_y = (nu*ep1_y + ep2_y)/(1-nu**2)
        tau_x = gamma_x/(2*(1+nu))
        tau_y = gamma_y/(2*(1+nu))
        
        #Now calculate the errors
        err_w = D*(wxxxx + wyyyy + 2*wxxyy)
        err_w += t**2 * txx *(nu*wyy + wxx) / (4*(1-nu**2))
        err_w += t**2 * tyy *(wyy + nu*wxx) / (4*(1-nu**2))
        err_w += t**2 * txy *(wxy) / (2*(1+nu))
        err_w += t * tx * (tx *(nu*wyy + wxx) + t*(wxyy + wxxx)) / (2*(1-nu**2))
        err_w += t * ty * (ty *(wyy + nu*wxx) + t*(wyyy + wxxy)) / (2*(1-nu**2))
        err_w -= t*sigx*(wxx + w0xx) + t*sigy*(1/r + wyy + w0yy) + 2*t*tau*(wxy + w0xy)

        err_u = t*sigx_x + tx*sigx + t*tau_y + ty*tau
        
        err_v = t*sigy_y + ty*sigy + t*tau_x + tx*tau
            
        #Set error for boundary conditions
        err_w[0,:] = w[1,:]
        err_w[1,:] = wx[1,:]
        err_w[-1,:] = w[-2,:]
        err_w[-2,:] = wx[-2,:]
        
        #err_v
        err_v[0,:] = v[1,:]
        err_v[-1,:] = v[-2,:]
        
        if bc_type == 'disp':
            #err_u
            err_u[0,:] = u[1,:]
            err_u[-1,:] = u[-2,:] - end_disp
        elif bc_type == 'load':
            err_u[0,:] = u[1,:]
            err_u[-1,0] = np.mean(sigx[-2,:]) - end_disp
            err_u[-1,1:] = u[-2,:-1] - u[-2,1:]
            
        return err_w, err_u, err_v
    
    def solve_single_disp(self, end_disp, err_marg = 1e-11, bc_type = 'disp', plotiter = 1e6, init = None):
        """
        Function uses a modified newton raphson method to determine the equilibrium deformations w, u, and v for the specified load. 
        The Jacobian is only updated when the error is no longer decreasing fast enough. 
        
        Arguments
          end_disp: End displacement or axial load on the shell
          err_marg: maximum error for which a solution is called an equilibrium
          bc_type: either 'disp' or 'load' for end displacement or axial load specified boundary conditions 
          plotiter: integer specifying how frequently to plot the current estimate for w, u, and v (useful for debugging)
          init: starting guess for the equilibrium provided in the form of a list [w_guess, u_guess, v_guess]
        Returns
          w: equilibrium deformations in the radial direction 
          u: equilibrium deformations in the axial direction
          v: equilibrium deformations in the circumferential direction
        """
        
        t0 = time.time()
        
        #If no initial guess is provided, use a starting guess of uniform radial expansion that one would get for a perfect shell
        if init is None:
            sigx0 = end_disp/self.L
            w = self.nu*sigx0*self.r*np.ones((self.Lpnts + 2, self.rpnts))
            u = np.meshgrid(np.ones(self.rpnts), np.linspace(-self.dx*sigx0, sigx0*self.L + sigx0*self.dx, self.Lpnts + 2))[1]
            v = np.zeros((self.Lpnts + 2, self.rpnts))
        else:
            w, u, v = init[0], init[1], init[2]

        #Create Jacobian calculating object
        # fastJacob = fastJacobian(self.r, self.L, self.nu, self.dgen, self.w0_derivs, self.t_derivs)
        fastJacob = self.fastJacob
        
        #Get maximum error associated with initial guess
        err_w, err_u, err_v = self.calcErrorCombined(w, u, v, end_disp, bc_type = bc_type)
        error_new = np.max(np.abs(np.concatenate((err_w, err_u, err_v))))
        
        print("Starting error: " + str(error_new))
        print("Creating and solving new Jacobian matrix.")
        
        i = 1
        error_old = 10*error_new #Initalize the old error as large, s.t. when entering the loop computing a new jacobian isn't prompted
 
        jacob = fastJacob.calcJacob(w, u, v, bc_type = bc_type) #calculate the Jacobian 
        jacob_solver = pypardiso.PyPardisoSolver() #Create a solver instance 
        
        while True:
            #If the new error isn't smaller than 3/4th of the previous, deem the convergence too slow and recompute the jacobian 
            if error_new > error_old*3/4:
                print("Creating and solving new Jacobian matrix.")
                del jacob #Remove the old jacobian from memory
                jacob = fastJacob.calcJacob(w, u, v, bc_type = bc_type)
            
            error_old = error_new
            
            #Perform an step of the newton raphson method
            dw, du, dv = self.stepJacobian(jacob, jacob_solver, err_w, err_u, err_v) 
            w, u, v = w+dw, u+du, v+dv
            
            #Calculate the new error
            err_w, err_u, err_v = self.calcErrorCombined(w, u, v, end_disp, bc_type = bc_type)
            error_new = np.max(np.abs(np.concatenate((err_w, err_u, err_v))))
            
            if isnotebook():
                clear_output(wait = True)
            print("Iteration " + str(i) + " error: " + str(error_new))

            if i%plotiter == 0: 
                print("Iteration " + str(i) + " plots")
                self.plotDisp(w, u, v)
                self.plotError(err_w, err_u, err_v)
                
            #If the new error is within the acceptable range, return w, u, and v
            if error_new < err_marg:
                print("Computing this load took " + str(time.time()-t0) + " seconds.")
                
                #Free the memory of the solver and delete the jacobian and the solver
                jacob_solver.free_memory(everything = True) 
                del jacob
                del jacob_solver
                
                break

            i += 1
        
        return w, u, v
    
    def solve_critical_disp(self, load_start = 0.1, dl = 0.1, dl_iter = 10, degmax = 1, err_marg = 1e-11, bc_type = 'disp', update_str = None):
        """
        Solve for the critical load of the shell.
        Function uses a modified newton raphson method to determine the equilibrium deformations w, u, and v for the specified load. 
        The Jacobian is only updated when the error is no longer decreasing fast enough. 
        
        Arguments
          load_start: the starting load as a fraction of the critical displacement or load expected for a perfect shell
          dl: the starting step size between loads checked as a fraction of the critical displacement or load. 
          dl_iter: the number of iterations on dl. Every iteration, the value of 'dl' is divided by 4.
          degmax: the maximum degree of the polynomial used to provide an interpolated initial estimate of w, u, v at each new load.
          end_disp: End displacement or axial load on the shell
          err_marg: maximum error for which a solution is called an equilibrium
          bc_type: either 'disp' or 'load' for end displacement or axial load specified boundary conditions
          update_str: string to be printed at every iteration (useful for debugging)
        Returns
          load_list: list of the loads for which equilibrium solutions were found 
          w_list: list of the equilibrium deformations in the radial direction associated with load_list
          u_list: list of the equilibrium deformations in the axial direction associated with load_list
          v_list: list of the equilibrium deformations in the circumferential direction associated with load_list
        """
        
        t00 = time.time()
        t0 = time.time()
        
        load_list = [0]
        Lpnts = self.Lpnts
        rpnts = self.rpnts
        dx = self.dx
        dy = self.dy
        
        w_list = [np.zeros((Lpnts + 2, rpnts))]
        u_list = [np.zeros((Lpnts + 2, rpnts))]
        v_list = [np.zeros((Lpnts + 2, rpnts))]
        
        r = self.r
        L = self.L
        nu = self.nu
        t = np.mean(self.t_derivs[0])
        
        #Failure theoretically is around sigma_x = -t/sqrt(3-3nu), end_disp = -L*t/sqrt(3-3nu)
        if bc_type == 'disp':
            load_cr = -L*t/(r*np.sqrt(3-3*nu**2))
        elif bc_type == 'load':
            load_cr = -t/(r*np.sqrt(3-3*nu**2))
        
        #Start iterating at this load
        load = load_start*load_cr
        #Load step size (this gets reduced whenever for a current load, the solution fails to convergence)
        dl = dl*load_cr
        #Minimum load step size
        dl_min = dl/(4**dl_iter)
        
        #Create Jacobian calculating object
        # fastJacob = fastJacobian(self.r, self.L, self.nu, self.dgen, self.w0_derivs, self.t_derivs)
        fastJacob = self.fastJacob
            
        #Initial guess for w, u, and v. Guessing uniform compression and expansion in w
        if bc_type == 'disp':
            sigx0 = load/L
        elif bc_type == 'load':
            sigx0 = load
            
        w = nu*sigx0*r*np.ones((Lpnts + 2, rpnts))
        u = np.meshgrid(np.ones(rpnts), np.linspace(-dx*sigx0, sigx0*L + sigx0*dx, Lpnts + 2))[1]
        v = np.zeros((Lpnts + 2, rpnts))
        
        #Get error associated with initial guess
        err_w, err_u, err_v = self.calcErrorCombined(w, u, v, load, bc_type = bc_type)
        error_new = np.max(np.abs(np.concatenate((err_w, err_u, err_v))))
        
        print("Starting error: " + str(error_new))
        print("Creating and solving new Jacobian matrix.")
        
        i = 1
        error_old = 10*error_new

        jacob = fastJacob.calcJacob(w, u, v, bc_type = bc_type)
        jacob_solver = pypardiso.PyPardisoSolver()
        
        new_jacob = False
        
        jacob_old = None
        jacob_solver_old = None
        
        while True:
            #If error is too large, one of two actions is taken:
            #1. If the Jacobian has not yet been recalculated for the current w, u, and v, recalculate it. the jacobian if it has not yet been calculated once for the current guess. 
            #2. Otherwise, reduce the load step size and try again for a lower load. If the step size is at its minimum, call the current load the buckling load. 
            if error_new > error_old*3/4:
                
                if not new_jacob:
                    #First, if this load was attempted without making a new jacobian try again by making a new jacobian
                    
                    print("Creating and solving new Jacobian matrix.")
                    if jacob_solver_old is not None:
                        jacob_solver_old.free_memory(everything = True)
                        del jacob_old 
                        del jacob_solver_old
                    
                    jacob_old = jacob
                    jacob_solver_old = jacob_solver
                    
                    #If the error went up, use w/u/v from the previous iteration. 
                    #This avoids problems if the latest iteration makes it much more wrong. 
                    if error_new > error_old:
                        w, u, v = w - dw, u - du, v - dv
                    
                    jacob = fastJacob.calcJacob(w, u, v, bc_type = bc_type)
                    
                    jacob_solver = pypardiso.PyPardisoSolver()
                    
                    error_old = 10
                    err_w, err_u, err_v = self.calcErrorCombined(w, u, v, load, bc_type = bc_type)
                    error_new  = np.max(np.abs(np.concatenate((err_w, err_u, err_v))))
                    
                    new_jacob = True
                elif dl/load_cr > 1.01*dl_min/load_cr:
                    #If with a new jacobian it fails to converge, then try again with a smaller step size
                    dl = dl/4
                    
                    load = load_list[-1] + dl
                    
                    print("Step size reduced to: " + str(dl/load_cr))
                    print("Load set to: " + str(load/load_cr))
                    
                    if degmax + 1 > len(load_list):
                        min_frame = 0
                    else:
                        min_frame = len(load_list) - (degmax + 1)
                    
                    wp, up, vp = getPoly1DMatrix(load_list[min_frame:], w_list[min_frame:], u_list[min_frame:], v_list[min_frame:])
                    w, u, v = predictDisp(load, wp, up, vp)
                    
                    #If the step size is still large, allow for the jacobian to be recalculated
                    #if dl_min >=  0.99*dl/(4**(5)):
                    new_jacob = False
                    
                    jacob_solver.free_memory(everything = True)
                    del jacob
                    del jacob_solver
                    
                    jacob = jacob_old
                    jacob_solver = jacob_solver_old
                    
                    error_old = 10
                    err_w, err_u, err_v = self.calcErrorCombined(w, u, v, load, bc_type = bc_type)
                    error_new  = np.max(np.abs(np.concatenate((err_w, err_u, err_v))))
                else:
                    #If the step size is below the minimum, associate the previous load with failure
                    #print("Failure load determined to be: " + str(load_list[-1]/load_cr))
                    print("Simulation complete. Took: " + str(time.time()-t00) + " seconds.")
                    
                    w = w_list[-1]
                    u = u_list[-1]
                    v = v_list[-1]
                    
                    jacob_solver_old.free_memory(everything = True)
                    jacob_solver.free_memory(everything = True)
                    
                    del jacob
                    del jacob_old
                    del jacob_solver
                    del jacob_solver_old
                    
                    break
                
            error_old = error_new
            
            dw, du, dv = self.stepJacobian(jacob, jacob_solver, err_w, err_u, err_v)
            w, u, v  = w + dw, u + du, v + dv
            
            err_w, err_u, err_v = self.calcErrorCombined(w, u, v, load, bc_type = bc_type)
            error_new = np.max(np.abs(np.concatenate((err_w, err_u, err_v))))
            
            if i%1 == 0:
                if isnotebook():
                    clear_output(wait = True)
                if update_str is not None:
                    print(update_str)
                print("Current load is: " + str(load/load_cr))
                print("Current dl iter is: " + str(dl_iter - np.round(np.log(dl/dl_min)/np.log(4))) + "/" + str(dl_iter)) #dl_min = dl/(4**dl_iter)
                print("Iteration " + str(i) + " error: " + str(error_new))
            
            if i%1e6 == 0: 
                print("Iteration " + str(i) + " plots")
                plotDisp(w, u, v)
                plotStress(w, u, v)
                plotError(err_w, err_u, err_v)
            
            if error_new < err_marg:

                t0 = time.time()
                
                load_list.append(load)
                
                w_list.append(w.copy())
                u_list.append(u.copy())
                v_list.append(v.copy())
                
                load = load_list[-1] + dl
                
                print("Load increased to: " + str(load/load_cr))
                
                if degmax + 1 > len(load_list):
                    min_frame = 0
                else:
                    min_frame = len(load_list) - (degmax + 1)
                
                wp, up, vp = getPoly1DMatrix(load_list[min_frame:], w_list[min_frame:], u_list[min_frame:], v_list[min_frame:])
                w, u, v = predictDisp(load, wp, up, vp)
                
                new_jacob = False
                
                error_old = 10
                err_w, err_u, err_v = self.calcErrorCombined(w, u, v, load, bc_type = bc_type)
                error_new  = np.max(np.abs(np.concatenate((err_w, err_u, err_v))))
                
                i = 0

            i += 1
        
        return load_list, w_list, u_list, v_list
    
    def stepJacobian(self, jacob, jacob_solver, err_w, err_u, err_v):
        """
        Performs a single step of the Newton-Raphson method specified by J(x_{n+1} - x_{n}) = -F(x_{n}), where J is the Jacobian, 
        x_{n} is a vector consisting of the deformations in the radial, axial and circumferential directions at iteration n. 
        x_{n+1} is the vector at the next iteration, which ought to be a closer approximation to the equilibrium. 
        F(x_{n}) is the error associated with the deformations specified by x_{n}. 
        
        Arguments
          jacob: the Jacobian matrix of the non-linear system of equations 
          jacob_solver: pypardiso sparse matrix solver object
          err_w: error in partial Energy partial w
          err_u: error in partial Energy partial u
          err_v: error in partial Energy partial v
          
        Returns
          dw: where w+dw is the next iteration estimate of the equilibrium radial deformations 
          du: where u+du is the next iteration estimate of the equilibrium axial deformations 
          dv: where v+dv is the next iteration estimate of the equilibrium circumferential deformations 
        """
        
        t0 = time.time()
        
        Lpnts = self.Lpnts
        rpnts = self.rpnts
        
        rhs = np.zeros(3*(Lpnts+2)*rpnts)
        rhs[:(Lpnts+2)*rpnts] = err_w.flatten()
        rhs[(Lpnts+2)*rpnts:2*(Lpnts+2)*rpnts] = err_u.flatten()
        rhs[2*(Lpnts+2)*rpnts:] = err_v.flatten()
        
        sol = pypardiso.spsolve(jacob, -rhs, solver = jacob_solver)
        #print("Took " + str(time.time()-t0) + " seconds to solve the jacobian.")
        return np.reshape(sol[0:(Lpnts+2)*rpnts], (Lpnts+2, rpnts)), np.reshape(sol[(Lpnts+2)*rpnts:2*(Lpnts+2)*rpnts], (Lpnts+2, rpnts)), np.reshape(sol[2*(Lpnts+2)*rpnts:], (Lpnts+2, rpnts))
    
    def calc_n_eigenmodes(self, w, u, v, n, bc_type = 'disp'):
        """
        Computes the n eigenmodes with the lowest eigenvalues using Arnoldi iteration for the configuration specified by w, u, and v. 
        
        Arguments
          w: deformations in the radial direction 
          u: deformations in the axial direction
          v: deformations in the circumferential direction
          n: dimension of Krylov subspace, must be >= 1
          bc_type: either 'disp' or 'load' for end displacement or axial load specified boundary conditions 
        
        Returns
          eigs: list of n eigenvalues
          eig_modes_w: list of radial deformation component of the n eigenmodes
          eig_modes_u: list of axial deformation component of the n eigenmodes
          eig_modes_v: list of circumferential deformation component of the n eigenmodes
        """
        
        #Create Jacobian calculating object
        # fastJacob = fastJacobian(self.r, self.L, self.nu, self.dgen, self.w0_derivs, self.t_derivs)
        fastJacob = self.fastJacob
        
        jacob = fastJacob.calcJacob(w, u, v, bc_type = bc_type)
        jacob_solver = pypardiso.PyPardisoSolver()
        
        t0 = time.time()
        
        Lpnts, rpnts = w.shape
        
        #Use a completely random initial test vector
        w0 = np.random.rand(Lpnts, rpnts)
        u0 = np.random.rand(Lpnts, rpnts)
        v0 = np.random.rand(Lpnts, rpnts)
        
        #Set BCs properly
        w0[0:2,:] = 0
        w0[-2:,:] = 0 
        
        w0x = self.dgen.deriv(w0, 1, 0)
        
        s_dx0 = self.dgen.gen_stencil(1,0,1)
        s_dxL = self.dgen.gen_stencil(1,0,Lpnts-2)
        
        w0[0,:] = -w0x[1,:]/s_dx0[0,int((s_dx0.shape[0]-1)/2)]
        w0[-1,:] = -w0x[-2,:]/s_dxL[-1,int((s_dxL.shape[0]-1)/2)]
        
        u0[1,:] = 0
        u0[-2,:] = 0 
        
        v0[1,:] = 0
        v0[-2,:] = 0 
        
        x0 = np.concatenate((w0.flatten(), u0.flatten(), v0.flatten()))
        
        Q, h = self.arnoldi_iter(jacob, jacob_solver, n, x0, w.shape)
        
        print("Arnoldi iteration done in time: ", time.time() - t0)
        
        h_sq = h[:-1, :]
        eigs, vecs = np.linalg.eig(h_sq)
        
        num_pnts = Lpnts*rpnts
        
        eig_modes_w = np.zeros((n, Lpnts, rpnts), dtype = np.complex128)
        eig_modes_u = np.zeros((n, Lpnts, rpnts), dtype = np.complex128)
        eig_modes_v = np.zeros((n, Lpnts, rpnts), dtype = np.complex128)
        
        for i in range(n):
            for j in range(n):
                eig_modes_w[i] += np.reshape(vecs[j,i]*Q[j,:][0:num_pnts], (Lpnts, rpnts))
                eig_modes_u[i] += np.reshape(vecs[j,i]*Q[j,:][num_pnts:2*num_pnts], (Lpnts, rpnts))
                eig_modes_v[i] += np.reshape(vecs[j,i]*Q[j,:][2*num_pnts:], (Lpnts, rpnts))
        
        #Sort the eigenmodes from largest to smallest eigenmode
        #Since we are finding the eigenmodes of the inverse matrix, the largest eigenvalue is the buckling mode
        sort = np.argsort(eigs)[::-1]
        
        eigs = 1/eigs[sort] #1/eigenvalues gives the eigenvalue of the original matrix instead of the inverse
        eig_modes_w = eig_modes_w[sort]
        eig_modes_u = eig_modes_u[sort]
        eig_modes_v = eig_modes_v[sort]
        
        #Free up memory
        jacob_solver.free_memory(everything = True)
        del jacob
        del jacob_solver
        
        return eigs, eig_modes_w, eig_modes_u, eig_modes_v
    
    def arnoldi_iter(self, A, jacob_solver, n, x0, dshape, eps = 1e-12):
        """
        Computes a basis of the (n + 1)-Krylov subspace of A inverse: the space
        spanned by {b, A^-1 b, ..., A^-n b}. This can be used to approximate the 
        n eigmodes with the lowest eigenvalues. 
        
        Arguments
          A: m × m sparse array
          jacob_solver: a pypardiso solver object 
          x0: initial vector (length m)
          n: dimension of Krylov subspace, must be >= 1
            
        Returns
          Q: m x (n + 1) array, the columns are an orthonormal basis of the
            Krylov subspace.
          h: (n + 1) x n array, A^-1 on basis Q. It is upper Hessenberg.  
        """
            
        Lpnts, rpnts = dshape
        h = np.zeros((n+1,n))
        Q = np.zeros((n+1, A.shape[0]))
        num_pnts = int(A.shape[0]/3)
        
        #Normalize the input vector
        Q[0,:] = x0/np.linalg.norm(x0,2)   # Use it as the first Krylov vector
        
        for k in range(1,n+1):
            # spsolve finds the solution to A x = rhs 
            # in other words, x = A^-1 rhs, so it can be considered just doing the product of the inverse of A with rhs. 
            # But, this avoids computing A^-1 directly, instead storing an LU-decomposition 
            
            rhs_w = np.reshape(Q[k-1,:].copy()[0:num_pnts], (Lpnts, rpnts))
            rhs_u = np.reshape(Q[k-1,:].copy()[num_pnts:2*num_pnts], (Lpnts, rpnts))
            rhs_v = np.reshape(Q[k-1,:].copy()[2*num_pnts:], (Lpnts, rpnts))
            #Add boundary conditions
            
            #w = 0 , wx = 0
            rhs_w[0:2,:] = 0
            rhs_w[-2:,:] = 0
            
            #u = end_disp
            rhs_u[0,:] = 0
            rhs_u[-1,:] = 0
            
            #v = 0
            rhs_v[0,:] = 0
            rhs_v[-1,:] = 0
            
            rhs = np.concatenate((rhs_w.flatten(), rhs_u.flatten(), rhs_v.flatten()))
            
            v = pypardiso.spsolve(A, rhs, solver = jacob_solver) #generate a new candidate vector
            
            for j in range(k):  # Subtract the projections on previous vectors
                h[j,k-1] = np.dot(Q[j,:].T, v)
                v = v - h[j,k-1] * Q[j,:]
            h[k,k-1] = np.linalg.norm(v,2)
            if h[k,k-1] > eps:  # Add the produced vector to the list, unless
                Q[k,:] = v/h[k,k-1]
            else:  # If that happens, stop iterating.
                return Q, h
            
        return Q, h
    
    def calc_single_eigenmode(self, w, u, v, n, eig_val = 0, bc_type = 'disp'):
        """
        Computes a single eigenmodes with eigenvalue close to the specified value using inverse power method. 
        
        Arguments
          w: deformations in the radial direction 
          u: deformations in the axial direction
          v: deformations in the circumferential direction
          n: number of iterations 
          eig_val: guess for the desired eigenvalue
          bc_type: either 'disp' or 'load' for end displacement or axial load specified boundary conditions 
        
        Returns
          eig_modes_w: list of radial deformation component of the eigenmode
          eig_modes_u: list of axial deformation component of the eigenmode
          eig_modes_v: list of circumferential deformation component of the eigenmode
        """
        
        jacob = self.fastJacob.calcJacob(w, u, v, bc_type = bc_type)
        jacob_solver = pypardiso.PyPardisoSolver()
        
        Lpnts, rpnts = w.shape
        
        num_pnts = Lpnts*rpnts
        
        x0 = np.random.rand(3*num_pnts)
        v = x0/np.linalg.norm(x0,2)
        for i in range(n):
            v = pypardiso.spsolve(jacob - eig_val*identity(jacob.shape[0]), v, solver = jacob_solver)
            v = v/np.linalg.norm(v,2)
        
        eig_mode_w = np.reshape(v[0:num_pnts], (Lpnts, rpnts))
        eig_mode_u = np.reshape(v[num_pnts:2*num_pnts], (Lpnts, rpnts))
        eig_mode_v = np.reshape(v[2*num_pnts:], (Lpnts, rpnts))
        
        jacob_solver.free_memory(everything = True)
        del jacob
        del jacob_solver
        
        return eig_mode_w, eig_mode_u, eig_mode_v
        
        
    def test_eigenmode(self, w, u, v, w_mode, u_mode, v_mode, eig_val = None, bc_type = 'disp'):
        """
        Tests the eigenvalue eigenmodes pairs computed with calc_n_eigenmodes or calc_single_eigenmode functions. 
        
        Arguments
          w: deformations in the radial direction 
          u: deformations in the axial direction
          v: deformations in the circumferential direction
          w_mode: radial deformation component of the eigenmode
          u_mode: axial deformation component of the eigenmode
          v_mode: circumferential deformation component of the eigenmode
          eig_val: eigenvalue of the eigenmode
          bc_type: either 'disp' or 'load' for end displacement or axial load specified boundary conditions 
        
        Outputs:
          subplot 1: radial deformation component of the eigenmode
          subplot 2: radial deformation component of J(x), where J is the jacobian and x is the three component vector eigenmode
          subplot 3: Difference between deformations in subplot 1 and subplot 2 / eig_val. This difference should approximately be zero. 
        """
       
        jacob = self.fastJacob.calcJacob(w, u, v, bc_type = bc_type)
        
        mode = np.concatenate((w_mode.flatten(), u_mode.flatten(), v_mode.flatten()))
        
        v = jacob @ mode
        Lpnts, rpnts = w.shape[0], w.shape[1]
        num_pnts = w.size
        
        eig_mode_w = np.reshape(v[0:num_pnts], (Lpnts, rpnts))
        eig_mode_u = np.reshape(v[num_pnts:2*num_pnts], (Lpnts, rpnts))
        eig_mode_v = np.reshape(v[2*num_pnts:], (Lpnts, rpnts))
        
        if eig_val is None:
            print(np.dot(mode, v))
            eig_val = np.dot(mode, v)
            
        plot.figure(figsize=(11,3.75), dpi= 100, facecolor='w', edgecolor='k')
        
        plt1 = plot.subplot(131)
        plt2 = plot.subplot(132)
        plt3 = plot.subplot(133)
        
        og = plt1.pcolor(w_mode[2:-2,:])
        plot.colorbar(og, ax = plt1)
        plt1.title.set_text('OG Eig')
        
        post = plt2.pcolor(eig_mode_w[2:-2,:]/eig_val)
        plot.colorbar(post, ax = plt2)
        plt2.title.set_text('One iter')
        
        diff = plt3.pcolor(eig_mode_w[2:-2,:]/eig_val - w_mode[2:-2,:])
        plot.colorbar(diff, ax = plt3)
        plt3.title.set_text('Diff')
        
        plot.show()
        
    def test_jacob(self, w, u, v, bc_type = 'disp', eps = 1e-6):
        """
        Function that checks whether or not the Jacobian was calculated correctly by comparing the results 
        from fastJacobian (done analytically and therefore faster but susceptible to coding errors) to a jacobian calculated
        numerically (much much slower, has numerical errors from finite difference methods, but easy to program)
        
        Arguments
          w: deformations in the radial direction 
          u: deformations in the axial direction
          v: deformations in the circumferential direction
          bc_type: either 'disp' or 'load' for end displacement or axial load specified boundary conditions 
          eps: finite difference step size for numerically computing the Jacobian 
          
        Outputs:
          subplot 1: fastJacobian output
          subplot 2: numerically computer Jacobian 
          subplot 3: their difference 
        """
        
        jacob = self.fastJacob.calcJacob(w, u, v, bc_type = bc_type)
        jacob_num = np.zeros(jacob.shape)
        
        def calc1DIndex(i, j,  disp, Lpnts, rpnts):
            #disp = 0 is w, 1 is u, 2 is v
            #calculates the 1D index from a 2D coordinates 
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
        
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                delta = np.zeros_like(w)
                delta[i,j] = eps
                
                err_w1, err_u1, err_v1 = self.calcErrorCombined(w+delta, u, v, 0)
                err_w2, err_u2, err_v2 = self.calcErrorCombined(w-delta, u, v, 0)
                jacob_num[calc1DIndex(i, j, 0, w.shape[0], w.shape[1]),:] = (1/eps/2)*np.concatenate(((err_w1 - err_w2).flatten(), (err_u1 - err_u2).flatten(), (err_v1 - err_v2).flatten()))
                
                err_w1, err_u1, err_v1 = self.calcErrorCombined(w, u+delta, v, 0)
                err_w2, err_u2, err_v2 = self.calcErrorCombined(w, u-delta, v, 0)
                jacob_num[calc1DIndex(i, j, 1, w.shape[0], w.shape[1]),:] = (1/eps/2)*np.concatenate(((err_w1 - err_w2).flatten(), (err_u1 - err_u2).flatten(), (err_v1 - err_v2).flatten()))
                
                err_w1, err_u1, err_v1 = self.calcErrorCombined(w, u, v+delta, 0)
                err_w2, err_u2, err_v2 = self.calcErrorCombined(w, u, v-delta, 0)
                jacob_num[calc1DIndex(i, j, 2, w.shape[0], w.shape[1]),:] = (1/eps/2)*np.concatenate(((err_w1 - err_w2).flatten(), (err_u1 - err_u2).flatten(), (err_v1 - err_v2).flatten()))
        
        plt1 = plot.subplot(221)
        plt2 = plot.subplot(222)
        plt3 = plot.subplot(223)
        
        plt_scl = 1e-2
        
        fast_map = plt1.pcolormesh(jacob.toarray(), vmin = -plt_scl, vmax = plt_scl)
        plot.colorbar(fast_map, ax = plt1)
        plt1.title.set_text('Fast Jacobian')
        
        num_map = plt2.pcolormesh(jacob_num.T, vmin = -plt_scl, vmax = plt_scl)
        plot.colorbar(num_map, ax = plt2)
        plt2.title.set_text('Numerical Jacobian')
        
        diff_map = plt3.pcolormesh(jacob_num.T - jacob, vmin = -plt_scl, vmax = plt_scl)
        plot.colorbar(diff_map, ax = plt3)
        plt3.title.set_text('Difference')
        
    
    def plotDisp(self, w, u, v):
        """
        Plots the radial, axial, and circumferential displacements. 
        Since axial displacements are dominated by a large linear compression term obfuscating smaller variations, 
        the first x derivative of u is plotted instead. 
        
        Arguments
          w: deformations in the radial direction 
          u: deformations in the axial direction
          v: deformations in the circumferential direction
        """
        
        X, Y, w0 = self.X, self.Y, self.w0_derivs[0]
        deriv = self.dgen.deriv
        
        plot.figure(figsize=(12, 6), dpi=80)
        plt1 = plot.subplot(221)
        plt2 = plot.subplot(222)
        plt3 = plot.subplot(223)

        w_map = plt1.pcolormesh(Y[1:-1,:], X[1:-1,:], w[1:-1,:], shading = 'auto')
        
        divider = make_axes_locatable(plt1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plot.colorbar(w_map, cax = cax)
        plt1.title.set_text(r'$w$')
        plt1.set_aspect(1)
        
        ux_map = plt2.pcolormesh(Y[1:-1,:], X[1:-1,:], deriv(u,1,0)[1:-1,:], shading = 'auto')
        
        divider = make_axes_locatable(plt2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plot.colorbar(ux_map, cax = cax)
        plt2.title.set_text(r'$u_x$')
        plt2.set_aspect(1)
        
        v_map = plt3.pcolormesh(Y[1:-1,:], X[1:-1,:], v[1:-1,:], shading = 'auto')
        
        divider = make_axes_locatable(plt3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plot.colorbar(v_map, cax = cax)
        plt3.title.set_text(r'$v$')
        plt3.set_aspect(1)
        
        plot.show()
    
    def plotImps(self):
        """
        Plots the radial and thickness geometric imperfections with which the solver object was initialized.
        """
        
        X, Y, w0, t0 = self.X, self.Y, self.w0_derivs[0], self.t_derivs[0]

        fig = plot.figure(figsize=(12, 6), dpi=80)
        plt1 = plot.subplot(121)
        plt2 = plot.subplot(122)

        w_map = plt1.pcolormesh(Y[1:-1,:], X[1:-1,:], w0[1:-1,:], shading = 'auto')
        
        divider = make_axes_locatable(plt1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plot.colorbar(w_map, cax = cax)
        
        plt1.title.set_text(r'$w_0$')
        plt1.set_aspect(1)
        
        ux_map = plt2.pcolormesh(Y[1:-1,:], X[1:-1,:], t0[1:-1,:], shading = 'auto')
        
        divider = make_axes_locatable(plt2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plot.colorbar(ux_map, cax = cax)
        
        plt2.title.set_text(r'$t$')
        plt2.set_aspect(1)
        
        plot.show()
    
    def plotStress(self, w, u, v):
        """
        Plots the axial, circumferential, and shear stresses for the specified deformations.
        
        Arguments
          w: deformations in the radial direction 
          u: deformations in the axial direction
          v: deformations in the circumferential direction
        """
        
        X, Y = self.X, self.Y
        
        sigx, sigy, tau = self.computeStresses(w, u, v)
        
        plot.figure(figsize=(12, 6), dpi=80)
        plt1 = plot.subplot(221)
        plt2 = plot.subplot(222)
        plt3 = plot.subplot(223)
        
        sigx_map = plt1.pcolormesh(Y[1:-1,:], X[1:-1,:], sigx[1:-1,:], shading = 'auto')
        
        divider = make_axes_locatable(plt1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plot.colorbar(sigx_map, cax = cax)
        
        plt1.title.set_text(r'$\sigma_x$')
        plt1.set_aspect(1)
        
        sigy_map = plt2.pcolormesh(Y[1:-1,:],X[1:-1,:], sigy[1:-1,:], shading = 'auto')
        
        divider = make_axes_locatable(plt2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plot.colorbar(sigy_map, cax = cax)
        
        plt2.title.set_text(r'$\sigma_y$')
        plt2.set_aspect(1)
        
        tau_map = plt3.pcolormesh(Y[1:-1,:], X[1:-1,:], tau[1:-1,:], shading = 'auto')
        
        divider = make_axes_locatable(plt3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plot.colorbar(tau_map, cax = cax)
        
        plt3.title.set_text(r'$\tau_{xy}$')
        plt3.set_aspect(1)
        
        plot.show()
        
    def plotError(self, err_w, err_u, err_v):
        """
        Plots the error. 
        
        Arguments
          err_w: error in partial Energy partial w
          err_u: error in partial Energy partial u
          err_v: error in partial Energy partial v
        """
        
        X, Y, w0 = self.X, self.Y, self.w0_derivs[0]
        
        plot.figure(figsize=(12, 6), dpi=80)
        plt1 = plot.subplot(221)
        plt2 = plot.subplot(222)
        plt3 = plot.subplot(223)

        err_map_w = plt1.pcolormesh(Y[1:-1,:], X[1:-1,:], err_w[1:-1,:], shading = 'auto')
        
        divider = make_axes_locatable(plt1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plot.colorbar(err_map_w, cax = cax)
        
        plt1.title.set_text(r'Error $w$')
        plt1.set_aspect(1)
        
        err_map_u = plt2.pcolormesh(Y[1:-1,:], X[1:-1,:], err_u[1:-1,:], shading = 'auto')
        
        divider = make_axes_locatable(plt2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plot.colorbar(err_map_u, cax = cax)
        
        plt2.title.set_text(r'Error $u$')
        plt2.set_aspect(1)
        
        err_map_v = plt3.pcolormesh(Y[1:-1,:], X[1:-1,:], err_v[1:-1,:], shading = 'auto')
        
        divider = make_axes_locatable(plt3)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plot.colorbar(err_map_v, cax = cax)
        
        plt3.title.set_text(r'Error $v$')
        plt3.set_aspect(1)
        
        plot.show()
    
    def saveCSV(self, ws, us, vs, name):
        """
        Saves the deformations associated with the simulations. 
        
        Arguments
          ws: list of the deformations in the radial direction to be saved
          us: list of the deformations in the axial direction  to be saved
          vs: list of the deformations in the circumferential direction  to be saved
          name: string of the folder name to be saved
          
        Outputs
          meta.csv: A meta data file, containg information about the simulation parameters
          imps.csv: An imperfections file, containing the geometric and thickness imperfections and their derivatives used in the simulation
          data.csv: A simulation data file, containing the deformations, stresses, and strains for every w, u, and v in ws, us, and vs. 
        """
        
        name = str(Path(sys.path[0]).parent.absolute()) + "\\Saved Simulations\\" + name
        
        Path(name).mkdir(parents = True, exist_ok = True)
        
        #First save the meta data file
        num_frames = len(ws)
        num_nodes = self.Lpnts*self.rpnts
        meta_dict = {'r': np.array([self.r]), 
                    'L': np.array([self.L]),
                    'nu': np.array([self.nu]),
                    'Lpnts': np.array([self.Lpnts]),
                    'rpnts': np.array([self.rpnts]),
                    'order': np.array([self.order]), 
                    'frames': np.array([num_frames])
                    }
        
        pandas.DataFrame(meta_dict).to_csv(name + "\\meta.csv", index = False)
        
        #Then save geometric imperfections
        empty_array = np.zeros(num_frames*num_nodes)
        
        imp_dict = {'w0': self.w0_derivs[0][1:-1,:].flatten(),
                    'w0x': self.w0_derivs[1][1:-1,:].flatten(),
                    'w0y': self.w0_derivs[2][1:-1,:].flatten(),
                    'w0xx': self.w0_derivs[3][1:-1,:].flatten(),
                    'w0xy': self.w0_derivs[4][1:-1,:].flatten(),
                    'w0yy': self.w0_derivs[5][1:-1,:].flatten(),
                    't': self.t_derivs[0][1:-1,:].flatten(),
                    'tx': self.t_derivs[1][1:-1,:].flatten(),
                    'ty': self.t_derivs[2][1:-1,:].flatten(),
                    'txx': self.t_derivs[3][1:-1,:].flatten(),
                    'txy': self.t_derivs[4][1:-1,:].flatten(),
                    'tyy': self.t_derivs[5][1:-1,:].flatten(),
                    }
        
        pandas.DataFrame(imp_dict).to_csv(name + "\\imps.csv", index = False)
        
        #Then save the simulation data
        data_dict = {'Frame': empty_array.copy(), 
                    'w': empty_array.copy(),
                    'u': empty_array.copy(),
                    'v': empty_array.copy(),
                    'ep1': empty_array.copy(),
                    'ep2': empty_array.copy(),
                    'gamma': empty_array.copy(),
                    'sigx': empty_array.copy(),
                    'sigy': empty_array.copy(),
                    'tau': empty_array.copy()
                    }
        
        for i in range(num_frames):
            w, u, v = ws[i], us[i], vs[i]
            ep1, ep2, gamma = self.computeStrains(w, u, v)
            sigx, sigy, tau = self.computeStresses(w, u, v)
            
            data_dict['Frame'][i*num_nodes:(i+1)*num_nodes] = i
            data_dict['w'][i*num_nodes:(i+1)*num_nodes] = w[1:-1,:].flatten()
            data_dict['u'][i*num_nodes:(i+1)*num_nodes] = u[1:-1,:].flatten()
            data_dict['v'][i*num_nodes:(i+1)*num_nodes] = v[1:-1,:].flatten()
            data_dict['ep1'][i*num_nodes:(i+1)*num_nodes] = ep1[1:-1,:].flatten()
            data_dict['ep2'][i*num_nodes:(i+1)*num_nodes] = ep2[1:-1,:].flatten()
            data_dict['gamma'][i*num_nodes:(i+1)*num_nodes] = gamma[1:-1,:].flatten()
            data_dict['sigx'][i*num_nodes:(i+1)*num_nodes] = sigx[1:-1,:].flatten()
            data_dict['sigy'][i*num_nodes:(i+1)*num_nodes] = sigy[1:-1,:].flatten()
            data_dict['tau'][i*num_nodes:(i+1)*num_nodes] = tau[1:-1,:].flatten()
        
        pandas.DataFrame(data_dict).to_csv(name + "\\data.csv", index = False)
        

def getPoly1DMatrix(load_list, w_list, u_list, v_list):
    num_frames, xpnts, ypnts = np.array(w_list).shape
    load_list = np.array(load_list, dtype = np.longdouble)
    
    ws = np.reshape(np.array(w_list, dtype = np.longdouble), (num_frames, xpnts*ypnts))
    us = np.reshape(np.array(u_list, dtype = np.longdouble), (num_frames, xpnts*ypnts))
    vs = np.reshape(np.array(v_list, dtype = np.longdouble), (num_frames, xpnts*ypnts))
    
    w = np.reshape(solveBulkVander(load_list, ws), (num_frames, xpnts, ypnts))
    u = np.reshape(solveBulkVander(load_list, us), (num_frames, xpnts, ypnts))
    v = np.reshape(solveBulkVander(load_list, vs), (num_frames, xpnts, ypnts))
    
    return w, u, v
    
def solveBulkVander(alpha, b):
    assert alpha.shape[0] == b.shape[0]
    
    n = b.shape[0]
    
    x = b.copy()

    for k in range(1, n):
        x[k:n,:] -= x[k - 1 : n - 1,:]
        x[k:n,:] /= alpha[k:n, np.newaxis] - alpha[0 : n - k, np.newaxis]

    for k in range(n - 1, 0, -1):
        x[k - 1 : n - 1,:] -= alpha[k - 1, np.newaxis] * x[k:n,:]

    return x

def predictDisp(load, wp, up, vp):
    deg = wp.shape[0]
    w = np.zeros((wp.shape[1], wp.shape[2]))
    u = np.zeros((wp.shape[1], wp.shape[2]))
    v = np.zeros((wp.shape[1], wp.shape[2]))
    
    for i in range(deg):
        w += wp[i,:,:]*load**i
        u += up[i,:,:]*load**i
        v += vp[i,:,:]*load**i
            
    return w, u, v
   
def predictEta(load, wp, up, vp):
    #Returns an estimate of dw/dload, du/dload and dv/dload at the input load 
    deg = wp.shape[0]
    eta_w = np.zeros((wp.shape[1], wp.shape[2]))
    eta_u = np.zeros((wp.shape[1], wp.shape[2]))
    eta_v = np.zeros((wp.shape[1], wp.shape[2]))
    
    for i in range(1, deg):
        eta_w += wp[i,:,:]*load**(i-1)
        eta_u += up[i,:,:]*load**(i-1)
        eta_v += vp[i,:,:]*load**(i-1)
            
    return eta_w, eta_u, eta_v
   
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
        
def defineDimpleImperfection(X, Y, r, L, t, nu, f, delta):
    #Perfect dimple imperfection at the center of the cylinder with amplitude delta and width f*halfwave
    #where the halfwave is defined using r, t, and nu
    
    halfwave = np.pi * np.sqrt(r*t / np.sqrt(12*(1-nu**2)))
    #print(halfwave)
    #Y = Y + np.pi*r
    
    w0 = delta*np.exp(-((X-L/2)**2 + (Y-np.pi*r)**2) / (f*halfwave)**2)
    w0x = delta*-2*(X-L/2)*np.exp(-((X-L/2)**2 + (Y-np.pi*r)**2) / (f*halfwave)**2)/(f*halfwave)**2
    w0xx = delta*4*((X-L/2)**2-(f*halfwave)**2/2)*np.exp(-((X-L/2)**2 + (Y-np.pi*r)**2) / (f*halfwave)**2)/(f*halfwave)**4
    w0y = delta*-2*(Y-np.pi*r)*np.exp(-((X-L/2)**2 + (Y-np.pi*r)**2) / (f*halfwave)**2)/(f*halfwave)**2
    w0yy = delta*4*((Y-np.pi*r)**2-(f*halfwave)**2/2)*np.exp(-((X-L/2)**2 + (Y-np.pi*r)**2) / (f*halfwave)**2)/(f*halfwave)**4
    w0xy = delta*4*(Y-np.pi*r)*(X-L/2)*np.exp(-((X-L/2)**2 + (Y-np.pi*r)**2) / (f*halfwave)**2)/(f*halfwave)**4
    
    w0_derivs = np.zeros((6, w0.shape[0], w0.shape[1]))
    w0_derivs[0] = w0
    w0_derivs[1] = w0x
    w0_derivs[2] = w0y
    w0_derivs[3] = w0xx
    w0_derivs[4] = w0xy
    w0_derivs[5] = w0yy
    return w0_derivs
    
