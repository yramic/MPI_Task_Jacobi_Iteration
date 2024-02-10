
Use "make" to build mpijacobi.cpp (prints error values and average runtimer per jacobi iteration in the console).
To then run with mpijacobi.cpp built, use "make run" or alternatively "mpirun -n #processes ./mpijacobi <2D/1D> <resolution> <iterations>".

The submission holds the following files:

- makefile:		compiles mpijacobi.cpp and enables a run of the programm with predefined input arguments
- solver.hpp: 		the heart of submission; this header file holdes both the 1D- and 2D-version MPI-Jacobi-solver-functions
- mpijacobi.cpp:	cpp file managing selection of solving method (1D vs 2D) and output
- functions.hpp:	header file containing functions for calculating error norms, the particular solution and other stand-alone computations

