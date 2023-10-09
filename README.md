# ShellNumerics

Cylindrical shells, e.g. the main component of rocket ships, buckle catastrophically when subjected to excessive loads. 
However, in real applications, their buckling loads vary significantly from one structure to the next, even when manufactured and tested with the exact same procedure. 
Determining exactly how large of a load a shell can sustain before failure is crucial to the safe and reliable applications of shells in engineering contexts.

Unfortunately, the buckling properties of shells are incredible sensitivity to non-uniformities in their radii and thickness, known as geometric imperfections, that inevitably and randomly occur during any manufacturing process. 
This imperfection sensitivity is so pronounced that developing experimental methods for predicting any individual real shell's buckling properties remains a domain of active research.
A necessary prerequisite for developing reliable predictive methods is having a qualitative and a quantitative understanding of the effect of imperfections.
The equations describing the dynamics of cylindrical shells---the von-Karman-Donnell equations---have been minimally investigated when accounting for realistic imperfection. 
In part, this is because they are a system of non-linear partial different equations, sensitive to boundary conditions, that cannot be solved analytically for generic imperfections. 

The software in this repository determines numerical solutions to the von Karman-Donnell equations for imperfect cylindrical shells with realistic boundary conditions.
It was used to analyze the properties of shells with real geometric imperfections characterized experimentally. 
The results of the numerical and experimental results, as well as the qualitative understanding developed therefrom, can be found in our paper.  

# Approach

Solve the non-linear system of equations resulting from discretizing the non-linear system of PDEs using a modified Newton-Raphson method.
\begin{align}
    \epsilon_1 &= u_x + \frac{1}{2}w_x^2 + w_x w_{0, x} \\ 
    \epsilon_2 &= v_y + \frac{w}{r} + \frac{1}{2}w_y^2 + w_y w_{0,y} \\
    \gamma &= v_x + u_y + w_x w_y + w_{0,x} w_y + w_x w_{0,y} 
\end{align}

Since a finite difference method discretization only couples adjacent points, the Jacobian for the system of equations is sparse. We therefore
use sparse matrices with an optimized sparse matrix solver (pypardiso) at each iteration of the Newton's method. 

Includes a single ghost point in the axial direction, and periodic boundary conditions in the circumferential direction. 

2D positions are represented as a 1D matrix where each element in the 1D matrix corresponds to an x,y coordinate
    #eigMatrix * w = rhs
    #w is a 1D matrix representation of the 2D space. x = -h, 0, h, ..., L with y= 0, then same thing with y=h, then y=2h, etc... 
Includes a "ghost point" on both sides in the x direction to encode the wxx = 0 boundary condition, such that there are Lpnts+2 elements in that direction 
"True" points are therefore from 1 to Lpnts when zero indexed
Ghost points are not required in the y direction since boundaries are periodic, these can just wrap around 

eigMatrix = np.zeros(((Lpnts+2)*rpnts, (Lpnts+2)*rpnts))
'''
