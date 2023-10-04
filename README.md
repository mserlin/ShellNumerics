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
