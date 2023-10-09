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

# von Karman-Donnell equations

In the formulation used by the von Karman-Donnell equations, the physical parameters are fully defined by the displacements of points along the shell's middle surface.
Variables are indexed by an $(x,y)$ coordinate, where $x$ refers to the axial position and $y$ the circumferential position.
For a shell of length $L$ and radius $r$, the axial position $x$ varies from $0$ to $L$, and the circumferential position $y$ varies from $0$ to $2\pi r$, where $y = 0$ and $y = 2\pi r$ are the same point.
Specifically, the $(x,y)$ coordinate refers to a point on the middle surface of the unloaded structure without imperfections: even when the structure deforms, the $(x,y)$ coordinate refers to the point on the structure that was originally at the specified position.
The deformation are captured by separate variables; in the axial direction, circumferential and radial directions the deformations are denoted $u(x,y)$, $v(x,y)$, and $w(x,y)$ respectively.
If the shell is imperfect and has variations in its curvature when unloaded, these are reflected by radial deformations $w_0(x,y)$.
The shell's thickness is given by $t(x,y)$, and, in generality, can vary spatially. 
The shell is taken to have a constant Poisson ratio of $\nu$ and Young's modulus of $E$. 
In the rest of this section, I use the shorthand, $w(x,y) = w$, $\frac{\partial w(x,y)}{\partial x} = w_x$ and $\frac{\partial^2 w(x,y)}{\partial x^2} = w_{xx}$, etc... 

The strain in the axial direction, $\epsilon_1$, circumferential direction, $\epsilon_2$, and shear strain $\gamma$ at the middle surface are given by:

$` \epsilon_1 = u_x + \frac{1}{2}w_x^2 + w_x w_{0, x} `$

$`\epsilon_2 = v_y + \frac{w}{r} + \frac{1}{2}w_y^2 + w_y w_{0,y} `$

$` \gamma = v_x + u_y + w_x w_y + w_{0,x} w_y + w_x w_{0,y} `$

where I have chosen the sign convention that outwards radial deformations are positive.

The stress in the axial direction, $\sigma_1$, circumferential direction, $\sigma_2$, and shear strain, $\tau$, at the middle surface are given by:

$`\sigma_1 = \frac{E}{1-\nu^2}\left(\epsilon_1 + \nu \epsilon_2 \right) `$

$`\sigma_2 = \frac{E}{1-\nu^2}\left(\epsilon_2 + \nu \epsilon_1 \right) `$

$`\tau = \frac{\gamma}{2(1+\nu)}`$

The von-Karman Donnell equations are the set of partial differential equations which must be satisfied in order for the deformations to represent an equilibrium solution. 
Typically, they are presented for a perfect shell with $w_0 = 0$ and a constant uniform thickness.
Relaxing these assumption, equations are given by:
$`\frac{E}{12(1-\nu^2)}\left(\frac{\partial^2}{\partial x^2}\left(t^3 (w_{xx} + \nu w_{yy})\right) + \frac{\partial^2}{\partial y^2}\left(t^3 (w_{yy} + \nu w_{xx})\right) +
2(1-\nu)\frac{\partial^2}{\partial x \partial y}\left(t^3 w_{xy}\right)\right) +
 t \sigma_x (w_{xx} + w_{0, xx}) + t \sigma_y (w_{yy} + w_{0, yy}) + 2 t \tau (w_{xy} + w_{0, xy}) = 0  `$

$` \frac{\partial}{\partial x}(t\sigma_x) + \frac{\partial}{\partial y}(t\tau) = 0 `$

$` \frac{\partial}{\partial y}(t\sigma_y) + \frac{\partial}{\partial x}(t\tau) = 0 `$

where all the variables in the above equations are functions of both $x$ and $y$. 

When there are no radial geometric imperfections, i.e. $w_0 = 0$, and the thickness is perfectly uniform such that $t(x,y) = t_{u}$, the equation simplifies to the cleaner, more familiar expression: 

$` \frac{Et_{u}^3}{12(1-\nu^2)}(w_{xxxx} + 2w_{xxyy} + w_{yyyy}) + t_{u} \sigma_x w_{xx} + t_{u}\sigma_y w_{yy} + 2 t_{u} \tau w_{xy} = 0 `$

Our boundary conditions are chosen such that the end of the shell maintains complete contact with the parallel loading plates at points held fixed by friction.
The friction requirement implies that the radial and circumferential displacements are zero at the end of the cylinder.
The contact requirement implies that the axial displacement at the ends are constant in the circumferential direction, and that the end of the shell remains normal to the loading plate.
The load on the shell is introduced by the value chosen for the end displacement, $\delta$, fixed by the parallel loading plates. 
The complete boundary conditions for a shell of length $L$ are written as: 

$` w(x = 0/L, y) = 0 `$

$` w_x(x = 0/L, y) = 0 `$

$` v(x = 0/L, y) = 0 `$

$` u(x = 0, y) = 0 \text{ and }  u(x = L, y) = \delta `$

# Numerical Methods

The equations are solved on a discretized two-dimensional rectangular grid with uniformed spacing in both the $x$ and $y$ direction, specified by $dx$ and $dy$. 
At each $(x,y)$ coordinate, the continuous derivatives are replaced by their finite difference approximation.
One-dimensional finite difference approximations of the derivatives are computed to arbitrary order using the Bjork-Pereyra algorithm, where an order of $N$ introduce errors that scale as $O(dx^{N})$ and $O(dy^{N})$.
Matrix representations of the two dimensional derivative operators, e.g. $\frac{\partial^2}{\partial x \partial y}$, are computed from outer products of the matrix representations of the one-dimensional operators.
Whenever possible, the finite difference approximations are central difference operators.
Near the ends of the cylinder, where $x = 0/L$ and the points required for a central difference approximation do not exist in both direction, I compute new finite difference operators that use more points, such that the accuracy of the approximation remains of same order.
The boundary conditions specified in the previous section are incorporated as additional constraints.

The discretization generates a non-linear system of equations whose solution approximates the shell's equilibrium deformations.
I numerically compute the solution using a modified Newton-Raphson method, where the Jacobian is calculated analytically. 
The Jacobian of the shell for the equilibrium deformations at a given end displacement, or load, is used to calculate the shell's eigenmodes with the smallest eigenvalues. 
Specifically, they are calculated using standard numerical techniques from a representation of the Jacobian's inverse in the basis of its Krylov subspace.

The buckling load is determined by taking advantage of the fact that, when the structure is unstable, the Newton-Raphson method fails to converge.
I slowly and incrementally increase the load, introduced by increasing $\delta$, until the a solution cannot be found. 
The largest load for which a stable solution can be found is a good approximation of the buckling load when the step size is small.

The code is written in python and uses Intel's Parallel Direct Sparse Solver Interface (PARDISO) to solve sparse matrices whenever required.

